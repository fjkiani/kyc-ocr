from document_classifier import DocumentClassifier
from mrz_processor import MRZProcessor
from llm_processor import LLMProcessor
import json
import numpy as np

def debug_print(message, data=None):
    print(f"\n🔍 DEBUG: {message}")
    if data:
        print(json.dumps(data, indent=2, cls=NumpyEncoder))

# Add a custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for NumPy data types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class DocumentProcessor:
    def __init__(self, api_key=None):
        debug_print("Initializing DocumentProcessor")
        self.classifier = DocumentClassifier()
        self.mrz_processor = MRZProcessor()
        self.llm_processor = LLMProcessor(api_key)
    
    def process_document(self, image_path):
        """Process a document image and extract relevant information"""
        debug_print("Starting document processing", {"image": image_path})
        
        # First classify the document
        classification = self.classifier.classify(image_path)
        debug_print("Document classification result", classification)
        
        result = {
            'document_type': classification['document_type'],
            'confidence': classification['confidence'],
            'detected_fields': classification['detected_fields'],
            'fields': {}
        }
        
        try:
            # Process with LLM
            debug_print("Starting LLM processing")
            llm_results = self.llm_processor.process_document_fields(
                classification['detected_fields'],
                classification['document_type']
            )
            debug_print("LLM processing results", llm_results)
            
            if llm_results:
                result['extracted_fields'] = llm_results['extracted_fields']
                result['validation_notes'] = llm_results['validation_notes']
                result['overall_confidence'] = llm_results['overall_confidence']
        except Exception as e:
            debug_print("Error in LLM processing", {"error": str(e), "type": type(e).__name__})
            import traceback
            print(traceback.format_exc())
        
        # If it's a passport, process MRZ and cross-validate with LLM results
        if classification['document_type'] == 'passport' and classification.get('has_mrz'):
            try:
                debug_print("Starting MRZ processing")
                mrz_data = self.mrz_processor.extract_mrz(image_path)
                if mrz_data:
                    result['mrz_data'] = mrz_data
                    # Cross-validate MRZ with visual text
                    result['cross_validation'] = self._cross_validate_mrz(
                        mrz_data, 
                        llm_results['extracted_fields'] if llm_results else {}
                    )
                debug_print("MRZ processing results", {"mrz_data": mrz_data})
            except Exception as e:
                debug_print("Error in MRZ processing", {"error": str(e), "type": type(e).__name__})
                import traceback
                print(traceback.format_exc())
        
        return result
    
    def _cross_validate_mrz(self, mrz_data, extracted_fields):
        """Cross-validate MRZ data with visually extracted fields"""
        validation_results = {
            'matches': [],
            'mismatches': [],
            'confidence': 0.0
        }
        
        # Compare fields that should match
        fields_to_compare = {
            'surname': 'last_name',
            'passport_number': 'document_number',
            'date_of_birth': 'dob',
            'expiry_date': 'expiry_date'
        }
        
        for mrz_field, visual_field in fields_to_compare.items():
            if mrz_field in mrz_data and visual_field in extracted_fields:
                if mrz_data[mrz_field] == extracted_fields[visual_field]['standardized_value']:
                    validation_results['matches'].append(mrz_field)
                else:
                    validation_results['mismatches'].append({
                        'field': mrz_field,
                        'mrz_value': mrz_data[mrz_field],
                        'visual_value': extracted_fields[visual_field]['standardized_value']
                    })
        
        # Calculate confidence based on matches
        total_fields = len(fields_to_compare)
        matched_fields = len(validation_results['matches'])
        validation_results['confidence'] = matched_fields / total_fields
        
        return validation_results 