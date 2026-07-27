import re
from typing import Dict, List, Tuple, Any, Union
from datasets import Dataset
from app.utils.logger import logger

class DatasetValidator:
    """Validates datasets checking for duplicates, empty values, invalid Unicode, and script mismatches."""
    
    # Unicode block ranges
    BENGALI_RANGE = re.compile(r'[\u0980-\u09FF]')
    MEITEI_MAYEK_RANGE = re.compile(r'[\uABC0-\uABFF\uAAE0-\uAAF6]')
    LATIN_RANGE = re.compile(r'[a-zA-Z]')

    def __init__(self, target_script: str = "any"):
        """
        Args:
            target_script (str): The expected script: 'bengali', 'meitei_mayek', 'roman', or 'any'.
        """
        self.target_script = target_script.lower()

    def is_invalid_unicode(self, text: str) -> bool:
        """Checks if the text contains invalid/corrupted Unicode characters."""
        # Check for replacement character indicating past decode error
        if "\uFFFD" in text:
            return True
        # Check for surrogate pairs or raw control chars (excluding spaces, tab, newline)
        if any(ord(char) < 32 and char not in "\n\r\t" for char in text):
            return True
        return False

    def is_script_mismatch(self, text: str) -> bool:
        """Checks if the text contains characters from the expected script."""
        if self.target_script == "any":
            return False
        
        has_bengali = bool(self.BENGALI_RANGE.search(text))
        has_meitei = bool(self.MEITEI_MAYEK_RANGE.search(text))
        has_latin = bool(self.LATIN_RANGE.search(text))

        if self.target_script == "bengali" and not has_bengali:
            return True
        if self.target_script == "meitei_mayek" and not has_meitei:
            return True
        if self.target_script == "roman" and not has_latin:
            return True
            
        return False

    def validate_samples(self, samples: List[Dict[str, Any]], text_keys: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Validates a list of dictionary samples.
        Args:
            samples: List of dictionaries to validate.
            text_keys: The keys in the dictionary containing text to be checked.
        Returns:
            Tuple of: (List of cleaned samples, dict containing the report count)
        """
        seen_sentences = set()
        cleaned_samples = []
        
        report = {
            "Loaded": len(samples),
            "Removed duplicates": 0,
            "Invalid Unicode": 0,
            "Empty": 0,
            "Script mismatch": 0,
            "Final": 0
        }

        for sample in samples:
            # Check if empty or null
            is_empty = False
            has_invalid_unicode = False
            has_script_mismatch = False
            is_duplicate = False

            # Extract fields to validate
            field_texts = []
            for key in text_keys:
                # Handle nested dicts (like HF translation key)
                val = sample.get(key)
                if isinstance(val, dict):
                    # check all values in translation dict
                    field_texts.extend([str(v) for v in val.values()])
                else:
                    field_texts.append(str(val) if val is not None else "")

            # Perform validation checks
            for text in field_texts:
                stripped = text.strip()
                if not stripped:
                    is_empty = True
                    break
                if self.is_invalid_unicode(text):
                    has_invalid_unicode = True
                    break
                if self.is_script_mismatch(text):
                    has_script_mismatch = True
                    break

            if is_empty:
                report["Empty"] += 1
                continue
            if has_invalid_unicode:
                report["Invalid Unicode"] += 1
                continue
            if has_script_mismatch:
                report["Script mismatch"] += 1
                continue

            # Duplicate detection on concatenated fields
            concat_fields = "||".join(field_texts)
            if concat_fields in seen_sentences:
                is_duplicate = True
            else:
                seen_sentences.add(concat_fields)

            if is_duplicate:
                report["Removed duplicates"] += 1
                continue

            # If all checks pass, keep the sample
            cleaned_samples.append(sample)

        report["Final"] = len(cleaned_samples)
        
        logger.info(
            f"Validation Report - Loaded: {report['Loaded']} | "
            f"Removed duplicates: {report['Removed duplicates']} | "
            f"Invalid Unicode: {report['Invalid Unicode']} | "
            f"Empty: {report['Empty']} | "
            f"Script mismatch: {report['Script mismatch']} | "
            f"Final: {report['Final']}"
        )
        return cleaned_samples, report

    def validate_dataset(self, dataset: Dataset, text_keys: List[str]) -> Tuple[Dataset, Dict[str, int]]:
        """
        Validates a HuggingFace Dataset object in-memory.
        """
        samples = [dict(row) for row in dataset]
        cleaned_samples, report = self.validate_samples(samples, text_keys)
        
        # Convert list of dicts back to HF Dataset
        if not cleaned_samples:
            empty_dict = {key: [] for key in dataset.features.keys()}
            cleaned_dataset = Dataset.from_dict(empty_dict)
        else:
            # Reconstruct list columns
            keys = cleaned_samples[0].keys()
            reconstructed = {key: [s[key] for s in cleaned_samples] for key in keys}
            cleaned_dataset = Dataset.from_dict(reconstructed, features=dataset.features)
            
        return cleaned_dataset, report
