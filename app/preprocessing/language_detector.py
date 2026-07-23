import logging
from typing import Dict, Any, List
from app.utils.logger import logger
from app.preprocessing.script_detector import ScriptDetector

class BaseLanguageDetector:
    """Abstract base class for pluggable language detectors."""
    def detect(self, text: str) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement detect()")


class LangdetectDetector(BaseLanguageDetector):
    """Language detector using the `langdetect` package."""
    def __init__(self):
        try:
            import langdetect
            self.detector = langdetect
            # Seed langdetect for deterministic results in tests
            self.detector.DetectorFactory.seed = 0
        except ImportError:
            logger.warning("langdetect package is not installed. LangdetectDetector will not work.")
            self.detector = None

    def detect(self, text: str) -> Dict[str, Any]:
        if not self.detector or not text.strip():
            return {"language": "unknown", "confidence": 0.0}
        
        try:
            probs = self.detector.detect_langs(text)
            best = probs[0]
            # Map common langdetect codes
            lang = best.lang
            return {"language": lang, "confidence": best.prob}
        except Exception:
            return {"language": "unknown", "confidence": 0.0}


class FastTextDetector(BaseLanguageDetector):
    """Language detector using Facebook's fastText model."""
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        try:
            import fasttext
            import os
            if os.path.exists(model_path):
                # Silence fasttext warnings
                fasttext.FastText.eprint = lambda x: None
                self.model = fasttext.load_model(model_path)
            else:
                logger.warning(f"FastText model not found at {model_path}. FastTextDetector is disabled.")
        except ImportError:
            logger.warning("fasttext package is not installed. FastTextDetector is disabled.")

    def detect(self, text: str) -> Dict[str, Any]:
        if not self.model or not text.strip():
            return {"language": "unknown", "confidence": 0.0}
        
        try:
            # Clean text of newlines for fasttext prediction
            cleaned = text.replace("\n", " ")
            labels, probabilities = self.model.predict(cleaned, k=1)
            if labels and probabilities:
                # fasttext labels are usually '__label__eng_Latn' or '__label__en'
                label = labels[0].replace("__label__", "")
                return {"language": label, "confidence": float(probabilities[0])}
        except Exception as e:
            logger.debug(f"FastText prediction error: {e}")
        return {"language": "unknown", "confidence": 0.0}


class LinguaDetector(BaseLanguageDetector):
    """Language detector using the `lingua` package."""
    def __init__(self):
        self.detector = None
        try:
            from lingua import Language, LanguageDetectorBuilder
            # Lingua supports ENGLISH, BENGALI, HINDI.
            # (Note: Lingua does not natively support Manipuri).
            self.languages = [Language.ENGLISH, Language.BENGALI, Language.HINDI]
            self.detector = LanguageDetectorBuilder.from_languages(*self.languages).build()
        except ImportError:
            logger.warning("lingua-py package is not installed. LinguaDetector is disabled.")

    def detect(self, text: str) -> Dict[str, Any]:
        if not self.detector or not text.strip():
            return {"language": "unknown", "confidence": 0.0}
        
        try:
            detected_lang = self.detector.detect_language_of(text)
            if detected_lang:
                # Map to standard ISO codes
                lang_map = {
                    "ENGLISH": "en",
                    "BENGALI": "bn",
                    "HINDI": "hi"
                }
                lang_code = lang_map.get(detected_lang.name, detected_lang.name.lower())
                confidence = self.detector.compute_language_confidence_values(text)
                prob = 0.0
                if confidence:
                    # Find confidence value for the detected language
                    for val in confidence:
                        if val.language == detected_lang:
                            prob = val.value
                            break
                return {"language": lang_code, "confidence": prob}
        except Exception as e:
            logger.debug(f"Lingua prediction error: {e}")
        return {"language": "unknown", "confidence": 0.0}


class LanguageDetector(BaseLanguageDetector):
    """
    Unified Language Detector.
    Uses configurable underlying detectors, and applies smart heuristics 
    specifically for Manipuri (mni) detection based on writing script.
    """
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        self.config = config
        self.enabled = config.get("enabled", True)
        self.detector_type = config.get("detector_type", "langdetect")
        self.min_confidence = config.get("min_confidence", 0.5)
        
        # Instantiate helper script detector for Manipuri heuristics
        self.script_detector = ScriptDetector()

        # Instantiate selected detector
        if self.detector_type == "fasttext":
            model_path = config.get("fasttext_model_path", "models/lid.176.ftz")
            self.base_detector = FastTextDetector(model_path)
        elif self.detector_type == "lingua":
            self.base_detector = LinguaDetector()
        else:
            self.base_detector = LangdetectDetector()

    def detect(self, text: str) -> Dict[str, Any]:
        """
        Detects language. Uses script detection first for high confidence 
        Manipuri Meitei Mayek detection.
        """
        if not self.enabled or not text:
            return {"language": "unknown", "confidence": 0.0}

        # Heuristic 1: If text is written in Meitei Mayek script, it is definitely Manipuri (mni)
        script_info = self.script_detector.detect(text)
        if script_info["script"] == "meitei" and script_info["confidence"] > 0.8:
            return {"language": "mni", "confidence": script_info["confidence"]}

        # Fallback to base detector (langdetect, fasttext, or lingua)
        result = self.base_detector.detect(text)

        # Heuristic 2: If the base detector returns 'bn' (Bengali) but the script is bengali, 
        # it could be either Bengali or Manipuri in Bengali script.
        # For simplicity, we keep the detected language or allow external mapping if configured.
        # But if the user target language is 'mni' and we detect 'bn', we can flag it for further verification.
        return result
