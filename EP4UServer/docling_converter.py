import torch

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_surya import SuryaOcrOptions
from docling.datamodel.accelerator_options import AcceleratorDevice


class DoclingParser:
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu and torch.cuda.is_available()

        self._build_pipeline()

        self.converter = self._build_converter()

    def _build_pipeline(self, gpu=True):
        if gpu:
            ocr_batch = 48
            layout_batch = 48
            device = AcceleratorDevice.CUDA
            use_gpu_flag = True
        else:
            ocr_batch = 8
            layout_batch = 8
            device = AcceleratorDevice.CPU
            use_gpu_flag = False

        self.pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            ocr_model="suryaocr",
            allow_external_plugins=True,
            ocr_options=SuryaOcrOptions(
                lang=["en"],
                force_full_page_ocr=True,
                use_gpu=use_gpu_flag,
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
            accelerator=device,
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
        try:
            result = self.converter.convert(path)
            return result.document

        except RuntimeError as e:
            if "out of memory" in str(e).lower() and self.use_gpu:
                print("GPU OOM -> switching to CPU w/ reduced batch size")

                torch.cuda.empty_cache()

                # rebuild pipeline for CPU (smaller batches)
                self._build_pipeline(gpu=False)
                self.converter = self._build_converter()

                result = self.converter.convert(path)
                return result.document

            raise
