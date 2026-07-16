from comfy_api.latest import io
import torch
from .pipeline import MiniT2IPipeline
from comfy.model_patcher import ModelPatcher
import comfy.model_management as mm
from transformers import T5EncoderModel # AutoTokenizer, 


model_downloaded = False


class MiniT2ITextEncoder(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniT2ITextEncoder",
            display_name="MiniT2I Text Encoder Loader",
            category="MiniT2I",
            inputs=[
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls) -> io.NodeOutput:
        load_device = mm.get_torch_device()
        offload_device = mm.intermediate_device()

        text_encoder = T5EncoderModel.from_pretrained(
             "google/flan-t5-large",
             torch_dtype=torch.float32,
             local_files_only=model_downloaded,
         )

        return io.NodeOutput(ModelPatcher(text_encoder, load_device, offload_device),)


class MiniT2ILoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniT2ILoader",
            display_name="MiniT2I Model Loader",
            category="MiniT2I",
            inputs=[
                io.Combo.Input("model_type", options=["b16","l16"], tooltip="l16=large, b16=normal"),
            ],
            outputs=[
                io.Model.Output(),
            ],
        )

    @classmethod
    def execute(cls, model_type) -> io.NodeOutput:
        transformer = MiniT2IPipeline.load_transformer(model_type)
        load_device = mm.get_torch_device()
        offload_device = mm.intermediate_device()

        return io.NodeOutput(ModelPatcher(transformer, load_device, offload_device),)


class MiniT2ISampler(io.ComfyNode):
    """
    An example node

    Class methods
    -------------
    define_schema (io.Schema):
        Tell the main program the metadata, input, output parameters of nodes.
    fingerprint_inputs:
        optional method to control when the node is re executed.
    check_lazy_status:
        optional method to control list of input names that need to be evaluated.

    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        """
            Return a schema which contains all information about the node.
            Some types: "Model", "Vae", "Clip", "Conditioning", "Latent", "Image", "Int", "String", "Float", "Combo".
            For outputs the "io.Model.Output" should be used, for inputs the "io.Model.Input" can be used.
            The type can be a "Combo" - this will be a list for selection.
        """
        return io.Schema(
            node_id="MiniT2I",
            display_name="MiniT2I Sampler",
            category="MiniT2I",
            inputs=[
                io.String.Input("prompt", multiline=True, lazy=True),
                io.Int.Input(
                    "steps",
                    default=10,
                    min=0,
                    max=4096,
                    step=1, # Slider's step
                    display_mode=io.NumberDisplay.number,  # Cosmetic only: display as "number" or "slider"
                    lazy=True,  # Will only be evaluated if check_lazy_status requires it
                ),
                io.Float.Input(
                    "guidance",
                    default=2.5,
                    min=0.0,
                    max=100.0,
                    step=0.1,
                    round=0.001, #The value representing the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                    display_mode=io.NumberDisplay.number,
                    lazy=True,
                ),
                io.Model.Input("model", tooltip="Use MiniT2I Loader", optional=True),
                io.Model.Input("text_encoder", tooltip="Use MiniT2I Text Encoder Loader", optional=True),
                io.Int.Input(
                    "seed",
                    default=1,
                    step=1, 
                    lazy=True,  # Will only be evaluated if check_lazy_status requires it
                ),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

#    @classmethod
#    def check_lazy_status(cls, image, string_field, int_field, float_field, print_to_screen):
#        """
#            Return a list of input names that need to be evaluated.
#
#            This function will be called if there are any lazy inputs which have not yet been
#            evaluated. As long as you return at least one field which has not yet been evaluated
#            (and more exist), this function will be called again once the value of the requested
#            field is available.
#
#            Any evaluated inputs will be passed as arguments to this function. Any unevaluated
#            inputs will have the value None.
#        """
#        if print_to_screen == "enable":
#            return ["int_field", "float_field", "string_field"]
#        else:
#            return []



    @classmethod
    def execute(cls, prompt, steps, guidance, model, text_encoder, seed) -> io.NodeOutput:
        global model_downloaded
        torch.manual_seed(seed)

        # transformer = model


        HUB_MODEL_ID = "MiniT2I/MiniT2I"
        pipe = MiniT2IPipeline.from_pretrained(
#        pipe = DiffusionPipeline.from_pretrained(
            HUB_MODEL_ID,
#            custom_pipeline=str(script_dir / "pipeline.py"),
            local_files_only=model_downloaded,
#            trust_remote_code=True,
        )
        if pipe:
            model_downloaded = True

        output = pipe(
            prompt,
            model.model, text_encoder.model,
#            model_type=model_type,
            model_dir=model.model.config._name_or_path.parent if model else None,
            guidance_scale=guidance,
            num_inference_steps=steps,
            torch_dtype=torch.bfloat16,
            local_files_only=model_downloaded,
            output_type="pt",
        )
 
        return io.NodeOutput(output.images,)

    """
        The node will always be re executed if any of the inputs change but
        this method can be used to force the node to execute again even when the inputs don't change.
        You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
        executed, if it is different the node will be executed again.
        This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
        changes between executions the LoadImage node is executed again.
    """
    #@classmethod
    #def fingerprint_inputs(s, image, string_field, int_field, float_field, print_to_screen):
    #    return ""

