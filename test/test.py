# SPDX-FileCopyrightText: © 2024 Tiny Tapeout
# SPDX-License-Identifier: Apache-2.0

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, Timer


@cocotb.test()
async def test_project(dut):
    dut._log.info("Start")

    # Set the clock period to 10 us (100 KHz)
    clock = Clock(dut.clk, 10, unit="us")
    cocotb.start_soon(clock.start())

    dut._log.info("Test project behavior")

    for i in range(40, -40, -1):
        await test_value(dut, i)


async def reset(dut):
    dut._log.info("Reset")
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut.rst_n.value = 1

async def test_value(dut, ui_in):
    expected_output = ui_in << 2
    expected_output += 128

    if ui_in <= -32:
        expected_output = 0
    elif ui_in >= 32:
        expected_output = 255

    dut._log.info(f"Test ui_in={ui_in}, expected uo_out={expected_output}")
    
    await reset(dut)

    dut.ui_in.value = ui_in
    #dut.uio_in.value = 0  # Dummy

    # Wait for one clock cycle to see the output values
    await ClockCycles(dut.clk, 1)

    # Wait to update registers
    await Timer(1, unit="ns") 

    assert dut.uo_out.value == expected_output, f"Expected uo_out to be {expected_output} for ui_in={ui_in}, but got {dut.uo_out.value}"

    dut._log.info(f"Passed for ui_in={ui_in}, received uo_out={dut.uo_out.value}")
