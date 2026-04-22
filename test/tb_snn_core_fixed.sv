`timescale 1ns/1ps

`ifndef INPUT_DIM
`define INPUT_DIM 784
`endif
`ifndef HIDDEN_DIM
`define HIDDEN_DIM 128
`endif
`ifndef OUTPUT_DIM
`define OUTPUT_DIM 10
`endif
`ifndef NUM_STEPS
`define NUM_STEPS 10
`endif
`ifndef BETA_NUM
`define BETA_NUM 983
`endif
`ifndef BETA_DEN
`define BETA_DEN 1024
`endif
`ifndef THRESHOLD
`define THRESHOLD 128
`endif

module tb_snn_core_fixed;
    reg clk = 1'b0;
    reg rst_n = 1'b0;
    reg start = 1'b0;
    wire done;

    integer fd;
    integer o;

    snn_core_fixed #(
        .INPUT_DIM(`INPUT_DIM),
        .HIDDEN_DIM(`HIDDEN_DIM),
        .OUTPUT_DIM(`OUTPUT_DIM),
        .NUM_STEPS(`NUM_STEPS),
        .BETA_NUM(`BETA_NUM),
        .BETA_DEN(`BETA_DEN),
        .THRESHOLD(`THRESHOLD),
        .W1_FILE("artifacts/ref_vectors_fixed/w1.memh"),
        .W2_FILE("artifacts/ref_vectors_fixed/w2.memh"),
        .SPIKES_FILE("artifacts/ref_vectors_fixed/spikes.memh")
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .done(done)
    );

    always #5 clk = ~clk;

    initial begin
        #20;
        rst_n = 1'b1;
        #10;
        start = 1'b1;
        #10;
        start = 1'b0;

        wait(done == 1'b1);
        fd = $fopen("artifacts/ref_vectors_fixed/verilog_logits.txt", "w");
        if (fd == 0) begin
            $display("ERROR: failed to open output file");
            $finish(1);
        end
        for (o = 0; o < `OUTPUT_DIM; o = o + 1) begin
            $fwrite(fd, "%0d\n", dut.logits[o]);
        end
        $fclose(fd);
        $display("Wrote artifacts/ref_vectors_fixed/verilog_logits.txt");
        $finish;
    end
endmodule
