`timescale 1ns/1ps

// Resource-first 4CH 640x640 AVM engine.
//
// The LUT and source frames are external memories.  This block deliberately
// uses one shared sampler and one source read request at a time; it does not
// infer the multi-megabyte LUT or any full-frame image buffer on chip.
// The memory-side contract is a one-request/one-response interface.  A real
// DDR wrapper may add burst/cache logic around this block without changing
// the fixed-point arithmetic or output contract.
module avm_4ch_640_stream #(
    parameter integer C_INPUT_WIDTH  = 640,
    parameter integer C_INPUT_HEIGHT = 320,
    parameter integer C_OUTPUT_WIDTH = 640,
    parameter integer C_OUTPUT_HEIGHT = 640,
    parameter integer C_LUT_ADDR_WIDTH = 19,
    parameter integer C_SRC_ADDR_WIDTH = 18
)(
    input  wire                         i_clk,
    input  wire                         i_rst_n,
    input  wire                         i_start,
    output reg                          o_busy,
    output reg                          o_frame_done,

    output reg                          o_lut_rd_en,
    output reg  [C_LUT_ADDR_WIDTH-1:0] o_lut_rd_addr,
    input  wire                         i_lut_rd_valid,
    input  wire [95:0]                  i_lut_rd_data,

    output reg                          o_src_rd_en,
    output reg  [1:0]                   o_src_rd_cam,
    output reg  [C_SRC_ADDR_WIDTH-1:0]  o_src_rd_addr,
    input  wire                         i_src_rd_valid,
    input  wire [15:0]                  i_src_rd_data,

    output reg                          o_pixel_valid,
    output reg  [15:0]                  o_pixel_rgb565,
    output reg  [C_LUT_ADDR_WIDTH-1:0]  o_pixel_index
);

    localparam [C_LUT_ADDR_WIDTH-1:0] C_LAST_PIXEL =
        C_OUTPUT_WIDTH * C_OUTPUT_HEIGHT - 1;

    localparam [3:0] S_IDLE       = 4'd0;
    localparam [3:0] S_LUT_REQ    = 4'd1;
    localparam [3:0] S_LUT_WAIT   = 4'd2;
    localparam [3:0] S_SRC_REQ    = 4'd3;
    localparam [3:0] S_SRC_WAIT   = 4'd4;
    localparam [3:0] S_SIDE_FINISH = 4'd5;
    localparam [3:0] S_OUTPUT     = 4'd6;

    reg [3:0] state;
    reg [C_LUT_ADDR_WIDTH-1:0] pixel_index;
    reg [1:0] neighbor_index;
    reg side_index;

    reg [C_SRC_ADDR_WIDTH-1:0] addr_a;
    reg [C_SRC_ADDR_WIDTH-1:0] addr_b;
    reg [7:0] fx_a;
    reg [7:0] fy_a;
    reg [7:0] fx_b;
    reg [7:0] fy_b;
    reg [10:0] alpha_q;
    reg [2:0] cam_a;
    reg [2:0] cam_b;
    reg entry_valid;
    reg entry_blend;

    reg [17:0] accum_r;
    reg [17:0] accum_g;
    reg [17:0] accum_b;
    reg [7:0] sampled_a_r;
    reg [7:0] sampled_a_g;
    reg [7:0] sampled_a_b;

    wire [7:0] selected_fx = side_index ? fx_b : fx_a;
    wire [7:0] selected_fy = side_index ? fy_b : fy_a;
    wire [2:0] selected_cam = side_index ? cam_b : cam_a;
    wire [C_SRC_ADDR_WIDTH-1:0] selected_addr = side_index ? addr_b : addr_a;

    function [C_SRC_ADDR_WIDTH-1:0] neighbor_addr;
        input [C_SRC_ADDR_WIDTH-1:0] base;
        input [1:0] n;
        begin
            case (n)
                2'd0: neighbor_addr = base;
                2'd1: neighbor_addr = base + {{(C_SRC_ADDR_WIDTH-1){1'b0}},1'b1};
                2'd2: neighbor_addr = base + C_INPUT_WIDTH;
                default: neighbor_addr = base + C_INPUT_WIDTH + 1;
            endcase
        end
    endfunction

    function [8:0] neighbor_weight;
        input [7:0] fx;
        input [7:0] fy;
        input [1:0] n;
        reg [8:0] one_minus_fx;
        reg [8:0] one_minus_fy;
        begin
            one_minus_fx = 9'd16 - {1'b0, fx};
            one_minus_fy = 9'd16 - {1'b0, fy};
            case (n)
                2'd0: neighbor_weight = one_minus_fx * one_minus_fy;
                2'd1: neighbor_weight = {1'b0, fx} * one_minus_fy;
                2'd2: neighbor_weight = one_minus_fx * {1'b0, fy};
                default: neighbor_weight = {1'b0, fx} * {1'b0, fy};
            endcase
        end
    endfunction

    function [15:0] pack_rgb565;
        input [7:0] r;
        input [7:0] g;
        input [7:0] b;
        begin
            pack_rgb565 = {r[7:3], g[7:2], b[7:3]};
        end
    endfunction

    wire [8:0] sample_weight = neighbor_weight(selected_fx, selected_fy, neighbor_index);
    wire [7:0] sample_r8 = {i_src_rd_data[15:11], i_src_rd_data[15:13]};
    wire [7:0] sample_g8 = {i_src_rd_data[10:5], i_src_rd_data[10:9]};
    wire [7:0] sample_b8 = {i_src_rd_data[4:0], i_src_rd_data[4:2]};
    wire [17:0] sample_product_r = sample_r8 * sample_weight;
    wire [17:0] sample_product_g = sample_g8 * sample_weight;
    wire [17:0] sample_product_b = sample_b8 * sample_weight;

    wire [7:0] side_r8 = (accum_r + 18'd128) >> 8;
    wire [7:0] side_g8 = (accum_g + 18'd128) >> 8;
    wire [7:0] side_b8 = (accum_b + 18'd128) >> 8;
    wire [21:0] blend_r_acc = alpha_q * sampled_a_r + (11'd64 - alpha_q) * side_r8;
    wire [21:0] blend_g_acc = alpha_q * sampled_a_g + (11'd64 - alpha_q) * side_g8;
    wire [21:0] blend_b_acc = alpha_q * sampled_a_b + (11'd64 - alpha_q) * side_b8;
    wire [7:0] blend_r8 = (blend_r_acc + 22'd32) >> 6;
    wire [7:0] blend_g8 = (blend_g_acc + 22'd32) >> 6;
    wire [7:0] blend_b8 = (blend_b_acc + 22'd32) >> 6;

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            state          <= S_IDLE;
            pixel_index    <= {C_LUT_ADDR_WIDTH{1'b0}};
            neighbor_index <= 2'd0;
            side_index     <= 1'b0;
            addr_a         <= {C_SRC_ADDR_WIDTH{1'b0}};
            addr_b         <= {C_SRC_ADDR_WIDTH{1'b0}};
            fx_a           <= 8'd0;
            fy_a           <= 8'd0;
            fx_b           <= 8'd0;
            fy_b           <= 8'd0;
            alpha_q        <= 11'd0;
            cam_a          <= 3'd7;
            cam_b          <= 3'd7;
            entry_valid    <= 1'b0;
            entry_blend    <= 1'b0;
            accum_r        <= 18'd0;
            accum_g        <= 18'd0;
            accum_b        <= 18'd0;
            sampled_a_r    <= 8'd0;
            sampled_a_g    <= 8'd0;
            sampled_a_b    <= 8'd0;
            o_busy         <= 1'b0;
            o_frame_done   <= 1'b0;
            o_lut_rd_en   <= 1'b0;
            o_lut_rd_addr <= {C_LUT_ADDR_WIDTH{1'b0}};
            o_src_rd_en   <= 1'b0;
            o_src_rd_cam  <= 2'd0;
            o_src_rd_addr <= {C_SRC_ADDR_WIDTH{1'b0}};
            o_pixel_valid <= 1'b0;
            o_pixel_rgb565 <= 16'd0;
            o_pixel_index <= {C_LUT_ADDR_WIDTH{1'b0}};
        end else begin
            o_lut_rd_en   <= 1'b0;
            o_src_rd_en   <= 1'b0;
            o_pixel_valid <= 1'b0;
            o_frame_done  <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (i_start) begin
                        o_busy      <= 1'b1;
                        pixel_index <= {C_LUT_ADDR_WIDTH{1'b0}};
                        state       <= S_LUT_REQ;
                    end
                end

                S_LUT_REQ: begin
                    o_lut_rd_en   <= 1'b1;
                    o_lut_rd_addr <= pixel_index;
                    state         <= S_LUT_WAIT;
                end

                S_LUT_WAIT: begin
                    if (i_lut_rd_valid) begin
                        addr_a      <= i_lut_rd_data[17:0];
                        addr_b      <= i_lut_rd_data[35:18];
                        fx_a        <= i_lut_rd_data[43:36];
                        fy_a        <= i_lut_rd_data[51:44];
                        fx_b        <= i_lut_rd_data[59:52];
                        fy_b        <= i_lut_rd_data[67:60];
                        alpha_q     <= i_lut_rd_data[78:68];
                        cam_a       <= i_lut_rd_data[81:79];
                        cam_b       <= i_lut_rd_data[84:82];
                        entry_valid <= i_lut_rd_data[85];
                        entry_blend <= i_lut_rd_data[86];
                        accum_r     <= 18'd0;
                        accum_g     <= 18'd0;
                        accum_b     <= 18'd0;
                        neighbor_index <= 2'd0;
                        side_index  <= 1'b0;
                        if (i_lut_rd_data[85]) begin
                            state <= S_SRC_REQ;
                        end else begin
                            o_pixel_rgb565 <= 16'd0;
                            state <= S_OUTPUT;
                        end
                    end
                end

                S_SRC_REQ: begin
                    o_src_rd_en   <= 1'b1;
                    o_src_rd_cam  <= selected_cam[1:0];
                    o_src_rd_addr <= neighbor_addr(selected_addr, neighbor_index);
                    state         <= S_SRC_WAIT;
                end

                S_SRC_WAIT: begin
                    if (i_src_rd_valid) begin
                        accum_r <= accum_r + sample_product_r;
                        accum_g <= accum_g + sample_product_g;
                        accum_b <= accum_b + sample_product_b;
                        if (neighbor_index == 2'd3) begin
                            state <= S_SIDE_FINISH;
                        end else begin
                            neighbor_index <= neighbor_index + 1'b1;
                            state <= S_SRC_REQ;
                        end
                    end
                end

                S_SIDE_FINISH: begin
                    if (!side_index) begin
                        if (entry_blend) begin
                            sampled_a_r <= side_r8;
                            sampled_a_g <= side_g8;
                            sampled_a_b <= side_b8;
                            accum_r <= 18'd0;
                            accum_g <= 18'd0;
                            accum_b <= 18'd0;
                            neighbor_index <= 2'd0;
                            side_index <= 1'b1;
                            state <= S_SRC_REQ;
                        end else begin
                            o_pixel_rgb565 <= pack_rgb565(side_r8, side_g8, side_b8);
                            state <= S_OUTPUT;
                        end
                    end else begin
                        o_pixel_rgb565 <= pack_rgb565(blend_r8, blend_g8, blend_b8);
                        state <= S_OUTPUT;
                    end
                end

                S_OUTPUT: begin
                    o_pixel_valid <= 1'b1;
                    o_pixel_index <= pixel_index;
                    if (pixel_index == C_LAST_PIXEL) begin
                        o_busy       <= 1'b0;
                        o_frame_done <= 1'b1;
                        state        <= S_IDLE;
                    end else begin
                        pixel_index <= pixel_index + 1'b1;
                        state       <= S_LUT_REQ;
                    end
                end

                default: begin
                    state <= S_IDLE;
                    o_busy <= 1'b0;
                end
            endcase
        end
    end
endmodule
