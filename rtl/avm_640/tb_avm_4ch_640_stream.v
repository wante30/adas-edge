`timescale 1ns/1ps

module tb_avm_4ch_640_stream;
    localparam integer OUT_PIXELS = 640 * 640;
    localparam integer SRC_PIXELS = 640 * 320;

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg start = 1'b0;
    wire busy;
    wire frame_done;
    wire lut_rd_en;
    wire [18:0] lut_rd_addr;
    reg lut_rd_valid = 1'b0;
    reg [95:0] lut_rd_data = 96'd0;
    wire src_rd_en;
    wire [1:0] src_rd_cam;
    wire [17:0] src_rd_addr;
    reg src_rd_valid = 1'b0;
    reg [15:0] src_rd_data = 16'd0;
    wire pixel_valid;
    wire [15:0] pixel_rgb565;
    wire [18:0] pixel_index;

    reg [95:0] lut_mem [0:OUT_PIXELS-1];
    reg [15:0] src0_mem [0:SRC_PIXELS-1];
    reg [15:0] src1_mem [0:SRC_PIXELS-1];
    reg [15:0] src2_mem [0:SRC_PIXELS-1];
    reg [15:0] src3_mem [0:SRC_PIXELS-1];
    reg [15:0] expected_mem [0:OUT_PIXELS-1];

    integer loaded;
    integer cycles;
    integer pixels;
    integer errors;
    integer max_cycles;
    reg [1023:0] lut_path;
    reg [1023:0] src0_path;
    reg [1023:0] src1_path;
    reg [1023:0] src2_path;
    reg [1023:0] src3_path;
    reg [1023:0] expected_path;

    avm_4ch_640_stream dut(
        .i_clk(clk), .i_rst_n(rst_n), .i_start(start), .o_busy(busy),
        .o_frame_done(frame_done), .o_lut_rd_en(lut_rd_en),
        .o_lut_rd_addr(lut_rd_addr), .i_lut_rd_valid(lut_rd_valid),
        .i_lut_rd_data(lut_rd_data), .o_src_rd_en(src_rd_en),
        .o_src_rd_cam(src_rd_cam), .o_src_rd_addr(src_rd_addr),
        .i_src_rd_valid(src_rd_valid), .i_src_rd_data(src_rd_data),
        .o_pixel_valid(pixel_valid), .o_pixel_rgb565(pixel_rgb565),
        .o_pixel_index(pixel_index)
    );

    always #5 clk = ~clk;

    initial begin
        if (!$value$plusargs("LUT=%s", lut_path) ||
            !$value$plusargs("SRC0=%s", src0_path) ||
            !$value$plusargs("SRC1=%s", src1_path) ||
            !$value$plusargs("SRC2=%s", src2_path) ||
            !$value$plusargs("SRC3=%s", src3_path) ||
            !$value$plusargs("EXPECTED=%s", expected_path)) begin
            $display("FAIL: required plusargs LUT/SRC0/SRC1/SRC2/SRC3/EXPECTED missing");
            $finish;
        end
        $display("Loading 640x640 AVM vectors...");
        $readmemh(lut_path, lut_mem);
        $readmemh(src0_path, src0_mem);
        $readmemh(src1_path, src1_mem);
        $readmemh(src2_path, src2_mem);
        $readmemh(src3_path, src3_mem);
        $readmemh(expected_path, expected_mem);
        loaded = 1;
    end

    // One-cycle response model for the external LUT memory.
    always @(posedge clk) begin
        lut_rd_valid <= lut_rd_en;
        if (lut_rd_en) lut_rd_data <= lut_mem[lut_rd_addr];
    end

    // One-cycle response model for the four external RGB565 frame memories.
    always @(posedge clk) begin
        src_rd_valid <= src_rd_en;
        if (src_rd_en) begin
            case (src_rd_cam)
                2'd0: src_rd_data <= src0_mem[src_rd_addr];
                2'd1: src_rd_data <= src1_mem[src_rd_addr];
                2'd2: src_rd_data <= src2_mem[src_rd_addr];
                default: src_rd_data <= src3_mem[src_rd_addr];
            endcase
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            cycles <= 0;
            pixels <= 0;
            errors <= 0;
        end else begin
            cycles <= cycles + 1;
            if (pixel_valid) begin
                pixels <= pixels + 1;
                if (pixel_rgb565 !== expected_mem[pixel_index]) begin
                    errors <= errors + 1;
                    if (errors < 10)
                        $display("MISMATCH index=%0d actual=%04h expected=%04h",
                            pixel_index, pixel_rgb565, expected_mem[pixel_index]);
                end
                if ((pixels + 1) % 100000 == 0)
                    $display("progress pixels=%0d cycles=%0d", pixels + 1, cycles);
            end
            if (cycles > 12000000) begin
                $display("FAIL: timeout cycles=%0d pixels=%0d errors=%0d", cycles, pixels, errors);
                $finish;
            end
        end
    end

    initial begin
        wait (loaded == 1);
        #100;
        @(posedge clk); rst_n <= 1'b1;
        @(posedge clk); start <= 1'b1;
        @(posedge clk); start <= 1'b0;
        wait (frame_done == 1'b1);
        // Let the monitor's nonblocking pixel/error counters settle before
        // checking the final frame count.
        @(posedge clk);
        @(negedge clk);
        if (pixels != OUT_PIXELS || errors != 0) begin
            $display("FAIL: pixels=%0d expected=%0d errors=%0d", pixels, OUT_PIXELS, errors);
        end else begin
            $display("PASS: 640x640 AVM RTL simulation pixels=%0d cycles=%0d", pixels, cycles);
        end
        $finish;
    end
endmodule
