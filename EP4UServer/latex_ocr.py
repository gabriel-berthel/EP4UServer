from typing import List
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.common.surya.schema import TaskNames
from typing import List
from PIL.Image import Image as PILImage

class SuryaLatexOCR():
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            foundation = FoundationPredictor()
            cls._instance.predictor = RecognitionPredictor(foundation)
        return cls._instance

    def run_single_block(self, images: List[PILImage]) -> List[str]:
        tasks = [TaskNames.block_without_boxes] * len(images)
        bboxes = [[[0, 0, img.width, img.height]] for img in images]

        for img in images:
            img.show()
        
        predictions = self.predictor(
            images,
            tasks,
            bboxes=bboxes,
        )

        return [
            p.text_lines[0].text if p.text_lines else ""
            for p in predictions
        ]