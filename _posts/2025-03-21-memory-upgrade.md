---
title: "노트북 RAM 증설하기"
# author:
#   name: Joung min Lim
#   link: https://github.com/min731
date: 2025-03-21 18:30:00 +0900
# categories: [AI | 딥러닝, Architecture]
# categories: [AI | 딥러닝, Concept]
# categories: [AI | 논문 리뷰, Attention is all you need]
# categories: [MLOps | 인프라 개발, Kserve]
categories: [Life | 일상 이야기, Trouble Shooting]
# categories: [STEM | 수학/통계, Statistics]
tags: [Lenovo-Legion-5, RAM, SO-DIMM]
description: "쿠버네티스 개발용 노트북의 램을 증설해봅시다."
image: assets/img/posts/resize/output/Samsung-1GB-DDR2-Laptop-RAM.jpg # 대표 이미지  가로 세로 비율 약 1.91:1 (예: 1200×628px)
math: true
toc: true
# pin: true
---

<div align="center">
  <small>Source: <a href="https://commons.wikimedia.org/wiki/File:Samsung-1GB-DDR2-Laptop-RAM.jpg">https://commons.wikimedia.org/wiki/File:Samsung-1GB-DDR2-Laptop-RAM.jpg</a></small>
</div>


> ## 개요

최근 실무에서 쿠버네티스 개발을 담당했습니다.

이에 따라 개인 시간에 노트북에 Minikube를 설치하여 공부하는 일이 잦아졌습니다.

![image-description](/assets/img/posts/resize/output/kubeflow-resourse-issue.png){: width="800px"}

하지만 제 **귀여운** 노트북의 스펙은 CPU 6 cores/12 threads, RAM 8 GB로 여러 third party들까지 배포하기에 매우 벅찬 환경이였습니다.

이에 따라 비어있는 2번째 램 슬롯에 램을 추가하여 보강하고자 합니다.

## 문제 해결

램 증설 방법은 생각보다 간단했습니다.

**(1) 노트북용 추가 램 구매**

삼성전자의 노트북용 DDR4 16GB을 구매하였습니다.

![image-description](/assets/img/posts/resize/output/buy-ram.png){: width="600px"}

**(2) 현재 램 상태 및 스펙 확인**

기존 메모리 8기가 메모리가 장착되어 있는 모습입니다.

![image-description](assets/img/posts/resize/input/additional-ram-before.png){: width="800px"}

```bash
jmlim@Legion-5:~$ sudo dmidecode -t memory | more
# dmidecode 3.3
Getting SMBIOS data from sysfs.
SMBIOS 3.2.0 present.

Handle 0x0022, DMI type 16, 23 bytes
Physical Memory Array
	Location: System Board Or Motherboard
	Use: System Memory
	Error Correction Type: None
	Maximum Capacity: 64 GB
	Error Information Handle: 0x0025
	Number Of Devices: 2

Handle 0x0023, DMI type 17, 84 bytes
Memory Device
	Array Handle: 0x0022
	Error Information Handle: 0x0026
	Total Width: 64 bits
	Data Width: 64 bits
	Size: 8 GB
	Form Factor: SODIMM
	Set: None
	Locator: DIMM 0
... more ...
```

- 'Number Of Devices: 2' -> 2개의 램 장착 가능
- 'Maximum Capacity: 64 GB' -> 최대 64 GB까지 증설 가능

**(3) 노트북 하판 분리**

노트북 하판의 나사를 풀어주고 하판을 분리해줍니다.

두번째 램 슬롯이 비어있습니다.

![image-description](/assets/img/posts/resize/input/notebook-before.jpg){: width="400px"}

**(4) 램 장착**

새로 구매한 램을 장착해줍니다.

![image-description](/assets/img/posts/resize/input/notebook-after.jpg){: width="400px"}

**(5) 증설된 램 상태 확인**

두번째 램 슬롯까지 추가로 장착되어 기존 8 GB의 메모리에서 24 GB로 늘어난 모습입니다.

![image-description](assets/img/posts/resize/output/additional-ram-after.png){: width="800px"}

```bash
jmlim@Legion-5:~$ sudo dmidecode -t memory | more
# dmidecode 3.3
Getting SMBIOS data from sysfs.
SMBIOS 3.2.0 present.

Handle 0x0022, DMI type 16, 23 bytes
Physical Memory Array
	Location: System Board Or Motherboard
	Use: System Memory
	Error Correction Type: None
	Maximum Capacity: 64 GB
	Error Information Handle: 0x0025
	Number Of Devices: 2

Handle 0x0023, DMI type 17, 84 bytes
Memory Device
	Array Handle: 0x0022
	Error Information Handle: 0x0026
	Total Width: 64 bits
	Data Width: 64 bits
	Size: 8 GB
	Form Factor: SODIMM
	Set: None
	Locator: DIMM 0
	Bank Locator: P0 CHANNEL A
	Type: DDR4
	Type Detail: Synchronous Unbuffered (Unregistered)
	Speed: 3200 MT/s
	Manufacturer: Kingston
	Serial Number: FD990E03
	Asset Tag: Not Specified
	Part Number: LV32D4S2S8HD-8      
	Rank: 1
	Configured Memory Speed: 3200 MT/s
	Minimum Voltage: 1.2 V
	Maximum Voltage: 1.2 V
	Configured Voltage: 1.2 V
	Memory Technology: DRAM
	Memory Operating Mode Capability: Volatile memory
	Firmware Version: Unknown
	Module Manufacturer ID: Bank 2, Hex 0x98
	Module Product ID: Unknown
	Memory Subsystem Controller Manufacturer ID: Unknown
	Memory Subsystem Controller Product ID: Unknown
	Non-Volatile Size: None
	Volatile Size: 8 GB
	Cache Size: None
	Logical Size: None

Handle 0x0024, DMI type 17, 84 bytes
Memory Device
	Array Handle: 0x0022
	Error Information Handle: 0x0027
	Total Width: 64 bits
	Data Width: 64 bits
	Size: 16 GB
	Form Factor: SODIMM
	Set: None
	Locator: DIMM 0
	Bank Locator: P0 CHANNEL B
	Type: DDR4
	Type Detail: Synchronous Unbuffered (Unregistered)
	Speed: 3200 MT/s
	Manufacturer: Samsung
	Serial Number: 374E4C4E
	Asset Tag: Not Specified
	Part Number: M471A2G43CB2-CWE    
	Rank: 1
	Configured Memory Speed: 3200 MT/s
	Minimum Voltage: 1.2 V
	Maximum Voltage: 1.2 V
	Configured Voltage: 1.2 V
	Memory Technology: DRAM
	Memory Operating Mode Capability: Volatile memory
	Firmware Version: Unknown
	Module Manufacturer ID: Bank 1, Hex 0xCE
	Module Product ID: Unknown
	Memory Subsystem Controller Manufacturer ID: Unknown
	Memory Subsystem Controller Product ID: Unknown
	Non-Volatile Size: None
	Volatile Size: 16 GB
	Cache Size: None
	Logical Size: None
```

## 마치며

램 추가 증설을 통해 앞으로 더욱 쾌적한 쿠버네티스 개발 환경을 구축할 수 있게 되었습니다.

감사합니다.