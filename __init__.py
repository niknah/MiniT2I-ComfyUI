from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
from .MiniT2I import MiniT2ISampler


class MiniT2IExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            MiniT2ISampler,
        ]


async def comfy_entrypoint() -> MiniT2IExtension:  # ComfyUI calls this to load your extension and its nodes.
    return MiniT2IExtension()

