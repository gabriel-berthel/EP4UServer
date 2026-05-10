import torch

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_surya import SuryaOcrOptions
from docling.datamodel.accelerator_options import AcceleratorDevice


class DoclingParser:
    def __init__(self, use_gpu=True):
        self.request_gpu = use_gpu
        self._build_pipeline(use_gpu=self._should_use_gpu())
        self.converter = self._build_converter()
        
    def _should_use_gpu(self):
        if not self.request_gpu:
            return False

        if not torch.cuda.is_available():
            return False

        free, _ = torch.cuda.mem_get_info()
        free_gb = free / (1024 ** 3)

        return free_gb > 2.0

    def _build_pipeline(self, use_gpu: bool):
        if use_gpu:
            self.accelerator = AcceleratorDevice.CUDA
            ocr_batch = 32
            layout_batch = 32
            ocr_gpu = True
        else:
            self.accelerator = AcceleratorDevice.CPU
            ocr_batch = 6
            layout_batch = 6
            ocr_gpu = False

        self.pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            ocr_model="suryaocr",
            allow_external_plugins=True,
            ocr_options=SuryaOcrOptions(
                lang=["en"],
                force_full_page_ocr=True,
                use_gpu=ocr_gpu,
            ),
            generate_picture_images=True,
            generate_table_images=True,
            generate_parsed_pages=True,
            do_formula_enrichment=True,
            do_picture_description=False,
            do_chart_extraction=False,
            do_picture_classification=False,
            images_scale=1.0,
            do_table_structure=True,
            accelerator=self.accelerator,
            ocr_batch_size=ocr_batch,
            layout_batch_size=layout_batch,
        )
        
    def _build_converter(self):
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.pipeline_options
                ),
            }
        )

    def parse(self, path):
        try:
            return self.converter.convert(path).document

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("GPU OOM detected → switching to CPU fallback")

                # aggressive cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

                # rebuild CPU pipeline
                self._build_pipeline(use_gpu=False)
                self.converter = self._build_converter()

                return self.converter.convert(path).document

            raise
