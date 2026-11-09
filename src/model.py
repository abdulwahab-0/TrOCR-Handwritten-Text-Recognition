import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


MODEL_NAME = "microsoft/trocr-small-stage1"


def load_trocr_model(model_name=MODEL_NAME):
    """
    Load the pretrained TrOCR processor and model.
    """

    print("Loading TrOCR processor...")

    processor = TrOCRProcessor.from_pretrained(model_name)

    print("Loading TrOCR model...")

    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    # Configure decoder token IDs
    model.config.decoder_start_token_id = (
        processor.tokenizer.cls_token_id
    )

    model.config.pad_token_id = (
        processor.tokenizer.pad_token_id
    )

    model.config.eos_token_id = (
        processor.tokenizer.sep_token_id
    )

    model.config.decoder.pad_token_id = (
        processor.tokenizer.pad_token_id
    )

    model.config.decoder.eos_token_id = (
        processor.tokenizer.sep_token_id
    )

    model.generation_config.decoder_start_token_id = (
        processor.tokenizer.cls_token_id
    )

    model.generation_config.pad_token_id = (
        processor.tokenizer.pad_token_id
    )

    model.generation_config.eos_token_id = (
        processor.tokenizer.sep_token_id
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)

    print("TrOCR model loaded successfully!")
    print("Device:", device)

    return processor, model, device
