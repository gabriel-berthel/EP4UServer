import os

allow_gpu = os.getenv("DOCLING_ALLOW_GPU", "False").strip().lower() == "true"

if not allow_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_surya import SuryaOcrOptions
import multiprocessing

class DoclingParser:
    def __init__(self):
        self.allow_gpu = allow_gpu

        self._build()
        self.converter = self._build_converter()

    def _build(self):
        batch = 16 if self.allow_gpu else 4

        self.pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            ocr_model="suryaocr",
            allow_external_plugins=True,
            ocr_options=SuryaOcrOptions(
                lang=["en"],
                force_full_page_ocr=True,
                use_gpu=self.allow_gpu,
            ),
            generate_picture_images=True,
            generate_table_images=True,
            generate_parsed_pages=True,
            do_formula_enrichment=False,
            do_picture_description=False,
            do_chart_extraction=False,
            do_picture_classification=False,
            images_scale=1.0,
            do_table_structure=True,
            accelerator=(
                AcceleratorDevice.CUDA if self.allow_gpu else AcceleratorDevice.CPU
            ),
            ocr_batch_size=batch,
            layout_batch_size=batch,
        )

        self.pipeline_options.table_structure_options.do_cell_matching = False

        self.pipeline_options.accelerator_options = AcceleratorOptions(
            device=(
                AcceleratorDevice.CUDA if self.allow_gpu else AcceleratorDevice.CPU
            ),
            num_threads=multiprocessing.cpu_count()
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
        return self.converter.convert(path).document
