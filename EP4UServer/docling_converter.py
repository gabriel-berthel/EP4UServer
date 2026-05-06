from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_surya import SuryaOcrOptions
from docling.datamodel.accelerator_options import AcceleratorDevice

class DoclingParser():
    # code enrichment => requires VLM
    def __init__(self):
        pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            ocr_model="suryaocr",
            allow_external_plugins=True,
            ocr_options=SuryaOcrOptions(lang=["en"], force_full_page_ocr=True, use_gpu=True),
            generate_picture_images=True,
            generate_table_images=True,
            generate_parsed_pages=True,
            do_formula_enrichment=True,
            do_picture_description=False,
            do_chart_extraction=False, # Requires granite
            do_picture_classification = False, # Requires Granite
            images_scale=1.0,
            do_table_structure=True,
            accelerator = AcceleratorDevice.CUDA,
            ocr_batch_size=48,
            layout_batch_size=48
        )

        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )

    def parse(self, path):
        convertion = self.converter.convert(path)
        return convertion.document
