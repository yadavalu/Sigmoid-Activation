/*
 * Copyright (c) 2025 Aadith Yadav Govindarajan
 * SPDX-License-Identifier: Apache-2.0
 */

`default_nettype none

module tt_um_sigmoid_8bit (
    input  wire [7:0] ui_in,    // Dedicated inputs
    output wire [7:0] uo_out,   // Dedicated outputs
    input  wire [7:0] uio_in,   // IOs: Input path
    output wire [7:0] uio_out,  // IOs: Output path
    output wire [7:0] uio_oe,   // IOs: Enable path (active high: 0=input, 1=output)
    input  wire       ena,      // always 1 when the design is powered, so you can ignore it
    input  wire       clk,      // clock
    input  wire       rst_n     // reset_n - low to reset
);

    // Pin Assignments
    // ui_in:  Input (x) in Q4.4 signed format
    // uo_out: Output (y) 0-1.0 scaled to 0-255

    // Tie off unused pins
    assign uio_out = 8'b0;
    assign uio_oe  = 8'b0;

    // Sigmoid function
    // We use a linear slope of 0.25 (y = 0.25x + 0.5)
    // Saturation: when 0.25x = +/- 0.5, ie x = +/- 2.0
    // In Q4.4, 2.0 = 32 (0010_0000)
    
    localparam signed [7:0] POS_SAT_LIMIT = 8'sd32;  // +2.0
    localparam signed [7:0] NEG_SAT_LIMIT = -8'sd32; // -2.0

    reg [7:0] y_reg;
    wire signed [7:0] x = ui_in; // Interpret input as signed

    always @(posedge clk) begin
        if (!rst_n) begin
            y_reg <= 8'd0;
        end else begin
            // Positive Saturation (x >= 2.0, y = 1)
            if (x >= POS_SAT_LIMIT) begin
                y_reg <= 8'd255;
            end 
            // Negative Saturation (x <= -2.0, y = 0)
            else if (x <= NEG_SAT_LIMIT) begin
                y_reg <= 8'd0;
            end 
            // Approximation: Linear Region (-2.0 < x < 2.0)
            // y = 0.25x + 0.5
            // Output = (Input * 4) + 128
            // Note: * 4 because Input Q4.4 (1.0 is 16), Output 0-255 (1.0 is 256). 
            // 256/16 = 16. 16 * 0.25 slope = 4.
            else begin
                y_reg <= (x << 2) + 8'd128;
            end
        end
    end

    // Assign registered output to pins
    assign uo_out = y_reg;

endmodule