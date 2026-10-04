/*
 * common_config.h
 *
 *  Created on: Nov 22, 2022
 *      Author: bigcat-himax
 */

#ifndef SCENARIO_TFLM_2IN1_FD_FL_PL_COMMON_CONFIG_H_
#define SCENARIO_TFLM_2IN1_FD_FL_PL_COMMON_CONFIG_H_

#define FRAME_CHECK_DEBUG 0
#define EN_ALGO 1
// #define SPI_SEN_PIC_CLK				(10000000)
#define SPI_SEN_PIC_CLK (12000000)

#define DBG_APP_LOG 0

// current FW image is 409600 bytes => 0x64000. set  0~0x171000 as FW area
#define FW_IMG_SZ 0x3A171000

// gesture model @ 0x3AB7B000 (xmodem flash offset = 0xB7B000)
//   file : model_zoo/tflm_yolov8_od/gesture_yolov8n_192_vela_0xB7B000.tflite
//   size : 2778912 bytes => 0x2A65E0
//   NOTE : Vela does NOT bake the flash address into the .tflite -- the
//          _0xB7B000 suffix is naming convention only. If this define is
//          changed, deploy.py FLASH_ADDR and build_and_flash.py MODEL_ADDR
//          must be changed to match, then re-flash BOTH firmware and model.
#define YOLOV8_OBJECT_DETECTION_FLASH_ADDR 0x3AB7B000

#endif /* SCENARIO_TFLM_2IN1_FD_FL_PL_COMMON_CONFIG_H_ */
