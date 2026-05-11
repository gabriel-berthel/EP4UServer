import time
import torch

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_surya import SuryaOcrOptions


class DoclingParser:
    gpu_disabled_until = 0

    def __init__(self, use_gpu=True, cooldown_seconds=300):
        self.request_gpu = use_gpu
        self.cooldown_seconds = cooldown_seconds

        self._build(use_gpu=self._gpu_ok())
        self.converter = self._build_converter()

    def _gpu_ok(self):
        if not self.request_gpu:
            return False
        if not torch.cuda.is_available():
            return False

        if time.time() < DoclingParser.gpu_disabled_until:
            return False

        free, _ = torch.cuda.mem_get_info()
        return (free / (1024 ** 3)) > 6.0

    def _batch_from_gpu(self):
        free, _ = torch.cuda.mem_get_info()
        gb = free / (1024 ** 3)

        if gb > 12:
            return 24
        if gb > 8:
            return 16
        if gb > 4:
            return 8
        return 4

    def _build(self, use_gpu):
        if use_gpu:
            accelerator = AcceleratorDevice.CUDA
            ocr_gpu = True
            batch = self._batch_from_gpu()
        else:
            accelerator = AcceleratorDevice.CPU
            ocr_gpu = False
            batch = 4

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
            ocr_batch_size=batch,
            layout_batch_size=batch,
        )

        self.pipeline_options.table_structure_options.do_cell_matching = True

        self.pipeline_options.accelerator_options = AcceleratorOptions(
            device=accelerator,
            num_threads=4
        )

        self.batch = batch
        self.use_gpu = use_gpu

    def _build_converter(self):
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.pipeline_options
                ),
            }
        )

    def _fallback_cpu(self):
        DoclingParser.gpu_disabled_until = time.time() + self.cooldown_seconds
        self._build(use_gpu=False)
        self.converter = self._build_converter()

    def parse(self, path):
        tries = 0

        while True:
            try:
                return self.converter.convert(path).document

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()

                    if self.use_gpu and tries == 0:
                        self._fallback_cpu()
                        tries += 1
                        continue

                raise
