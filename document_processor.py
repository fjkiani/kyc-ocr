class DocumentProcessor:
    def __init__(self, api_key=None):
        self.classifier = DocumentClassifier()
        self.mrz_processor = MRZProcessor()
        self.llm_processor = LLMProcessor(api_key)
    
    def process_document(self, image_path):
        """Process a document image and extract relevant information"""
        # First classify the document
        classification = self.classifier.classify(image_path)
        
        result = {
            'document_type': classification['document_type'],
            'confidence': classification['confidence'],
            'fields': {}
        }
        
        # Process with LLM
        llm_results = self.llm_processor.process_document_fields(
            classification['detected_fields'],
            classification['document_type']
        )
        
        if llm_results:
            result['extracted_fields'] = llm_results['extracted_fields']
            result['validation_notes'] = llm_results['validation_notes']
            result['overall_confidence'] = llm_results['overall_confidence']
        
        # If it's a passport, process MRZ and cross-validate with LLM results
        if classification['document_type'] == 'passport' and classification['has_mrz']:
            mrz_data = self.mrz_processor.extract_mrz(image_path)
            if mrz_data:
                result['mrz_data'] = mrz_data
                # Cross-validate MRZ with visual text
                result['cross_validation'] = self._cross_validate_mrz(
                    mrz_data, 
                    llm_results['extracted_fields']
                )
        
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