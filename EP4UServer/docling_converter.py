import torch

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_surya import SuryaOcrOptions
from docling.datamodel.accelerator_options import AcceleratorDevice


class DoclingParser:
    def __init__(self, use_gpu=True):
        self.request_gpu = use_gpu
        self._build_pipeline(self._gpu_is_safe())
        self.converter = self._build_converter()

    def _gpu_is_safe(self):
        if not self.request_gpu:
            return False
        if not torch.cuda.is_available():
            return False
        free, _ = torch.cuda.mem_get_info()
        return (free / (1024 ** 3)) > 8.0

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

    def parse(self, path):
        return self.converter.convert(path).document
