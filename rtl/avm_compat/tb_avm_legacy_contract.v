`timescale 1ns/1ps

// Reference simulation only; no claim about a user's board DDR adapter.
module tb_avm_legacy_contract;
    parameter OUT_W = 480;
    parameter OUT_H = 640;
    localparam OUT_PIXELS = OUT_W * OUT_H;
    localparam SRC_PIXELS = 640 * 320;
    reg clk=0, rst_n=0, start=0;
    wire busy, done, lut_en, src_en, pixel_valid;
    wire [18:0] lut_addr, pixel_index;
    wire [17:0] src_addr;
    wire [1:0] src_cam;
    wire [15:0] pixel_data;
    reg lut_valid=0, src_valid=0;
    reg [95:0] lut_data=0;
    reg [15:0] src_data=0;
    reg [95:0] lut_mem [0:OUT_PIXELS-1];
    reg [15:0] src0 [0:SRC_PIXELS-1];
    reg [15:0] src1 [0:SRC_PIXELS-1];
    reg [15:0] src2 [0:SRC_PIXELS-1];
    reg [15:0] src3 [0:SRC_PIXELS-1];
    reg [15:0] expected [0:OUT_PIXELS-1];
    reg [2047:0] lut_path,p0,p1,p2,p3,expected_path;
    integer cycles=0, pixels=0, errors=0;

    avm_4ch_640_stream #(.C_OUTPUT_WIDTH(OUT_W), .C_OUTPUT_HEIGHT(OUT_H)) dut (
        .i_clk(clk),.i_rst_n(rst_n),.i_start(start),.o_busy(busy),.o_frame_done(done),
        .o_lut_rd_en(lut_en),.o_lut_rd_addr(lut_addr),.i_lut_rd_valid(lut_valid),.i_lut_rd_data(lut_data),
        .o_src_rd_en(src_en),.o_src_rd_cam(src_cam),.o_src_rd_addr(src_addr),
        .i_src_rd_valid(src_valid),.i_src_rd_data(src_data),
        .o_pixel_valid(pixel_valid),.o_pixel_rgb565(pixel_data),.o_pixel_index(pixel_index));
    always #5 clk=~clk;
    always @(posedge clk) begin
        lut_valid<=lut_en;
        src_valid<=src_en;
        if(lut_en) begin
            if(lut_addr>=OUT_PIXELS) $fatal(1,"LUT address out of range");
            lut_data<=lut_mem[lut_addr];
        end
        if(src_en) begin
            if(src_addr>=SRC_PIXELS) $fatal(1,"Source address out of range");
            case(src_cam)
                0:src_data<=src0[src_addr];
                1:src_data<=src1[src_addr];
                2:src_data<=src2[src_addr];
                3:src_data<=src3[src_addr];
            endcase
        end
        if(rst_n) begin
            cycles<=cycles+1;
            if(cycles>OUT_PIXELS*64) $fatal(1,"Simulation cycle limit");
            if(pixel_valid) begin
                if(pixel_index!==pixels) $fatal(1,"Output order mismatch");
                pixels<=pixels+1;
                if(pixel_data!==expected[pixel_index]) begin
                    errors<=errors+1;
                    if(errors<5) $display("MISMATCH index=%0d actual=%04h expected=%04h",pixel_index,pixel_data,expected[pixel_index]);
                end
            end
        end
    end
    initial begin
        if(!$value$plusargs("LUT=%s",lut_path)||!$value$plusargs("SRC0=%s",p0)||
           !$value$plusargs("SRC1=%s",p1)||!$value$plusargs("SRC2=%s",p2)||
           !$value$plusargs("SRC3=%s",p3)||!$value$plusargs("EXPECTED=%s",expected_path))
            $fatal(1,"Missing memory file arguments");
        $readmemh(lut_path,lut_mem);$readmemh(p0,src0);$readmemh(p1,src1);
        $readmemh(p2,src2);$readmemh(p3,src3);$readmemh(expected_path,expected);
        #100;
        @(posedge clk);rst_n<=1;
        @(posedge clk);start<=1;
        @(posedge clk);start<=0;
        wait(done);
        @(posedge clk);@(negedge clk);
        if(pixels!=OUT_PIXELS||errors!=0) $fatal(1,"FAIL pixels=%0d errors=%0d",pixels,errors);
        $display("PASS: legacy contract %0dx%0d pixels=%0d mismatches=%0d cycles=%0d",OUT_W,OUT_H,pixels,errors,cycles);
        $finish;
    end
endmodule
