import time
import torch

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_surya import SuryaOcrOptions
from docling.datamodel.accelerator_options import AcceleratorDevice


class DoclingParser:
    def __init__(self, use_gpu=True, cooldown_seconds=300):
        self.request_gpu = use_gpu
        self.cooldown_seconds = cooldown_seconds
        self.gpu_disabled_until = 0

        self._build_pipeline(use_gpu=self._should_use_gpu())
        self.converter = self._build_converter()

    def _gpu_is_available(self):
        if not self.request_gpu:
            return False
        if not torch.cuda.is_available():
            return False

        now = time.time()
        if now < self.gpu_disabled_until:
            return False

        free, _ = torch.cuda.mem_get_info()
        return (free / (1024 ** 3)) > 4.0

    def _build_pipeline(self, use_gpu):
        if use_gpu:
            accelerator = AcceleratorDevice.CUDA
            ocr_batch = 24
            layout_batch = 24
            ocr_gpu = True
        else:
            accelerator = AcceleratorDevice.CPU
            ocr_batch = 4
            layout_batch = 4
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
            accelerator=accelerator,
            ocr_batch_size=ocr_batch,
            layout_batch_size=layout_batch,
        )

        self.pipeline_options.table_structure_options.do_cell_matching = True

    def _build_converter(self):
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.pipeline_options
                ),
            }
        )

    def _switch_to_cpu_and_cooldown(self):
        self.gpu_disabled_until = time.time() + self.cooldown_seconds
        self._build_pipeline(use_gpu=False)
        self.converter = self._build_converter()

    def _should_use_gpu(self):
        return self._gpu_is_available()

    def parse(self, path):
        try:
            return self.converter.convert(path).document

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

                self._switch_to_cpu_and_cooldown()
                return self.converter.convert(path).document

            raise
