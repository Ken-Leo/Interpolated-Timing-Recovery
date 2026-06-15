# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Implementation of the Oversampled Per-Survivor Processing Viterbi Algorithm (OPSP-VA) for timing recovery in magnetic recording channels.

## Architecture

- `src/channel/`: Signal generation, pulse shapes (Longitudinal/Perpendicular), and jitter models.
- `src/frontend/`: 7th-order Butterworth LPF and Fractionally Spaced Equalizer (FSE).
- `src/opsps_va/`: Core Viterbi loop, per-survivor PLLs, and Early-Late TED.
- `src/utils/`: BER and convergence metrics.
- `tests/`: Unit and integration tests for each module.

## Common Commands

- Run all tests: `pytest tests/`
- Run a specific test: `pytest tests/test_<module>.py`
- Run simulation: `python main.py`

## Implementation Notes

- Target PR Code: PR-IV ($H(D) = 1 - D^2$).
- Language: Python 3.10+
- Key Libraries: `numpy`, `scipy`.

## Rules

- 禁止在项目之外创建和保存数据，只允许在项目所在文件夹中进行数据组织

- 及时维护进展报告、实施计划、技术报告的markdown文件

- 各环节结果可视化成图保存

## References

- 光盘仿真系统，含回读模型、数字均衡、LMS自适应均衡、Viterbi检测、trellis构建等。目录：/Volumes/Elements/MyPy/BD_Sim_for_BERvsiMLSE

- 磁盘dibit response提取仿真系统，含LMR和PMR回读模型、dibit response提取过程，支持1.25倍采样，也可以实现更高采样。目录：/Volumes/Elements/MyPy/dibit

- MATLAB 仿真插值算法的位同步技术，以BPSK调制数据为对象，包含了内插滤波器模块、定时误差计算模块、环路滤波器模块等。目录：/Volumes/Elements/PSP/project/E8_11_gardner.m

- 一维 PRML 仿真系统，包含 回读模型、GPR均衡器系数和GPR Target计算、均衡、检测、NLTS计算模拟和补偿等。目录：/Volumes/Elements/PRML-dynamicNLTS/python_receiver
