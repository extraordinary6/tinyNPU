// dff.sv — minimal D flip-flop; smoke test for cocotb/Icarus/py37 toolchain.
// Ports: clk, rst_n (sync active-low), d, q.

module dff (
    input  logic clk,
    input  logic rst_n,
    input  logic d,
    output logic q
);

    always_ff @(posedge clk) begin
        if (!rst_n) q <= 1'b0;
        else        q <= d;
    end

endmodule
