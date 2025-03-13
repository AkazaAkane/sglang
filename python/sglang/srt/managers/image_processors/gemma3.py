import asyncio
import math
from typing import List, Union

from PIL import Image

from sglang.srt.managers.image_processor import BaseImageProcessor
from sglang.srt.managers.image_processors.base_image_processor import (
    get_global_processor,
)
from sglang.srt.models.gemma3 import Gemma3ForCausalLM
from sglang.srt.models.gemma3_mm import Gemma3ForConditionalGeneration


class Gemma3ImageProcessor(BaseImageProcessor):
    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)
        self.IMAGE_TOKEN = "<image_soft_token"

    @staticmethod
    def _process_images_task(images, input_text, _hf_config):
        if isinstance(images, list) and len(images) == 0:
            images = None
        result = get_global_processor().__call__(
            text=[input_text], images=images, padding=True, return_tensors="pt"
        )

        return {
            "input_ids": result.input_ids,
            "pixel_values": getattr(result, "pixel_values", None),
            "image_grid_thw": getattr(result, "image_grid_thw", None),
            "second_per_grid_ts": getattr(result, "second_per_grid_ts", None),
            "video_grid_thws": getattr(result, "video_grid_thws", None),
        }

    async def _process_images(self, images, input_text) -> dict:
        if self.executor is not None:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor,
                Gemma3ImageProcessor._process_images_task,
                images,
                input_text,
                self.hf_config,
            )
        else:
            return self._process_images_task(images, input_text, self.hf_config)

    async def process_images_async(
        self,
        image_data: List[Union[str, bytes]],
        input_ids,
        request_obj,
        max_req_input_len,
        *args,
        **kwargs,
    ):
        if not image_data:
            return None
        if isinstance(image_data, str):
            image_data = [image_data]

        image_token = self.IMAGE_TOKEN
        base_output = self.load_images(
            input_ids,
            image_data,
            image_token,
            max_req_input_len,
        )

        ret = await self._process_images(base_output.input_text, base_output.input_text)
        return {
            "input_ids": ret["input_ids"].flatten().tolist(),
            "pixel_values": ret["pixel_values"],
            "image_hashes": base_output.image_hashes,
            "modalities": request_obj.modalities or ["image"],
            "image_grid_thws": ret["image_grid_thw"],
            "video_grid_thws": ret["video_grid_thws"],
            "im_start_id": self.IM_START_TOKEN_ID,
            "im_end_id": self.IM_END_TOKEN_ID,
            "im_token_id": self.image_token_id,
            "video_token_id": self.video_token_id,
            "second_per_grid_ts": ret["second_per_grid_ts"],
        }


ImageProcessorMapping = {
    Gemma3ForConditionalGeneration: Gemma3ImageProcessor,
}
