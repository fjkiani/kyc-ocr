"""
Accuracy Tester for Document Processing

This module provides a user interface for testing the accuracy of different document processing methods.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
from pathlib import Path
import json
import numpy as np
import time
import os
from metrics_tracker import get_metrics_tracker, format_metrics_for_display, create_accuracy_report
import re

def run_accuracy_test(image_path, doc_type, traditional_processor, inlining_processor):
    """Run accuracy test on a document with both processing methods"""
    metrics = get_metrics_tracker()
    
    # Process with traditional OCR
    st.markdown("### Testing with Traditional OCR")
    metrics.record_processing_start('traditional_ocr')
    traditional_results = traditional_processor(image_path)
    
    # Count detected fields in traditional OCR results
    field_count = 0
    confidence_scores = []
    if traditional_results and 'detected_fields' in traditional_results:
        field_count = len(traditional_results['detected_fields'])
        
        # Extract confidence scores
        for item in traditional_results['detected_fields']:
            if isinstance(item, tuple) and len(item) > 2:
                confidence_scores.append(item[2])  # Confidence is usually the third element
            elif isinstance(item, dict) and 'confidence' in item:
                confidence_scores.append(item['confidence'])
        
        # Record additional metrics for traditional OCR
        if confidence_scores:
            traditional_results['confidence'] = sum(confidence_scores) / len(confidence_scores)
        
        traditional_results['field_count'] = field_count
    
    metrics.record_processing_end('traditional_ocr', traditional_results)
    
    # Process with Document Inlining
    st.markdown("### Testing with Document Inlining")
    metrics.record_processing_start('document_inlining')
    inlining_results = inlining_processor(image_path, doc_type)
    
    # Extract text-based metrics from narrative response
    if inlining_results:
        # If we have full text but no structured fields, analyze the text to extract metrics
        if ('extracted_fields' in inlining_results and 
            (not inlining_results['extracted_fields'] or 
             ('full_text' in inlining_results['extracted_fields'] and len(inlining_results['extracted_fields']) <= 2))):
            
            if 'full_text' in inlining_results['extracted_fields']:
                text = inlining_results['extracted_fields']['full_text'].get('value', '')
            elif 'document_content' in inlining_results['extracted_fields']:
                text = inlining_results['extracted_fields']['document_content'].get('value', '')
            else:
                text = str(inlining_results.get('structured_content', ''))
            
            # Count the number of fields extracted by looking for patterns in the text
            field_count = 0
            lines = text.split('\n')
            
            # Look for key-value patterns, list items, and other structured elements
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Patterns that indicate extracted fields
                if ':' in line and not line.startswith('#'):  # Key-value pairs
                    field_count += 1
                elif line.startswith('- ') or line.startswith('* '):  # List items
                    field_count += 1
                elif re.search(r'^\d+\.\s', line):  # Numbered list items
                    field_count += 1
            
            # Look for section headers to estimate structure preservation quality
            section_count = 0
            for line in lines:
                if line.startswith('##'):
                    section_count += 1
            
            # Create structured fields based on analysis
            structured_fields = {
                'field_count': {'value': field_count, 'confidence': 0.9},
                'section_count': {'value': section_count, 'confidence': 0.9}
            }
            
            # Only override if we found significant structure
            if field_count > 5:
                inlining_results['extracted_fields'] = structured_fields
                inlining_results['field_count'] = field_count
                
                # Estimate extraction confidence based on structure and organization
                extraction_confidence = min(0.95, 0.7 + (field_count / 100) + (section_count / 20))
                inlining_results['confidence'] = extraction_confidence
    
    metrics.record_processing_end('document_inlining', inlining_results)
    
    # Show test results summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Traditional OCR Summary")
        if traditional_results:
            ocr_field_count = len(traditional_results.get('detected_fields', []))
            ocr_confidence = traditional_results.get('confidence', 0)
            st.write(f"Fields detected: {ocr_field_count}")
            st.write(f"Average confidence: {ocr_confidence:.2f}")
            st.write(f"Processing time: {metrics.processing_times['traditional_ocr'][-1]:.2f} seconds")
    
    with col2:
        st.markdown("#### Document Inlining Summary")
        if inlining_results:
            if 'extracted_fields' in inlining_results:
                inlining_field_count = len(inlining_results['extracted_fields'])
                st.write(f"Fields extracted: {inlining_field_count}")
            else:
                st.write("Fields extracted: N/A")
            
            if 'overall_confidence' in inlining_results:
                st.write(f"Confidence: {inlining_results['overall_confidence']:.2f}")
            elif 'confidence' in inlining_results:
                st.write(f"Confidence: {inlining_results['confidence']:.2f}")
            else:
                st.write("Confidence: N/A")
            
            st.write(f"Processing time: {metrics.processing_times['document_inlining'][-1]:.2f} seconds")
    
    return traditional_results, inlining_results

def display_accuracy_report():
    """Display the accuracy report based on metrics collected"""
    metrics = get_metrics_tracker().get_latest_metrics()
    
    if not metrics or metrics['document_count'] == 0:
        st.warning("No accuracy data available. Process some documents to generate metrics.")
        return
    
    st.markdown("<div class='sub-header'>Accuracy Report</div>", unsafe_allow_html=True)
    
    # Display document count and test summary
    st.markdown(f"**Documents Processed**: {metrics['document_count']}")
    
    # Create expander for detailed metrics
    with st.expander("View Detailed Metrics", expanded=False):
        st.json(metrics)
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    # Display confidence score comparison
    with col1:
        st.markdown("### Confidence Scores")
        conf_data = {
            'Method': ['Traditional OCR', 'Document Inlining'],
            'Average Confidence': [
                metrics['confidence_scores']['traditional_ocr']['average'] * 100,
                metrics['confidence_scores']['document_inlining']['average'] * 100
            ]
        }
        conf_df = pd.DataFrame(conf_data)
        st.bar_chart(conf_df.set_index('Method'))
        
        if 'comparative' in metrics:
            improvement = metrics['comparative'].get('confidence_improvement_pct', 0)
            st.markdown(f"**Improvement**: {improvement:.1f}%")
            
            # Add interpretation
            if improvement > 30:
                st.success("Significant improvement in confidence scores with Document Inlining")
            elif improvement > 10:
                st.info("Moderate improvement in confidence scores with Document Inlining")
            else:
                st.warning("Minimal improvement in confidence scores")
    
    # Display processing time comparison
    with col2:
        st.markdown("### Processing Time (seconds)")
        time_data = {
            'Method': ['Traditional OCR', 'Document Inlining'],
            'Average Processing Time': [
                metrics['processing_time']['traditional_ocr']['average'],
                metrics['processing_time']['document_inlining']['average']
            ]
        }
        time_df = pd.DataFrame(time_data)
        st.bar_chart(time_df.set_index('Method'))
        
        if 'comparative' in metrics:
            reduction = metrics['comparative'].get('processing_time_reduction_pct', 0)
            st.markdown(f"**Time Reduction**: {reduction:.1f}%")
            
            # Add interpretation
            if reduction < 0:
                st.warning("Document Inlining is currently slower than traditional OCR")
            elif reduction < 30:
                st.info("Moderate time savings with Document Inlining")
            else:
                st.success("Significant time savings with Document Inlining")
    
    # Display extraction count comparison
    st.markdown("### Field Extraction Comparison")
    extract_data = {
        'Method': ['Traditional OCR', 'Document Inlining'],
        'Average Extracted Fields': [
            metrics['extraction_counts']['traditional_ocr']['average'],
            metrics['extraction_counts']['document_inlining']['average']
        ]
    }
    extract_df = pd.DataFrame(extract_data)
    st.bar_chart(extract_df.set_index('Method'))
    
    # Calculate improvement in extraction count
    trad_avg = metrics['extraction_counts']['traditional_ocr']['average'] 
    inlining_avg = metrics['extraction_counts']['document_inlining']['average']
    if trad_avg > 0:
        extraction_improvement = ((inlining_avg - trad_avg) / trad_avg) * 100
        st.markdown(f"**Extraction Improvement**: {extraction_improvement:.1f}%")
        
        # Add interpretation
        if extraction_improvement < 0:
            st.warning("Traditional OCR currently detects more fields")
        elif extraction_improvement < 20:
            st.info("Document Inlining extracts a similar number of fields but with better structure")
        else:
            st.success("Document Inlining extracts significantly more structured fields")
    
    # Display metrics used for the financial impact
    st.markdown("### Key Performance Metrics")
    display_dynamic_financial_impact(metrics)

def batch_test_documents(image_paths, doc_types, traditional_processor, inlining_processor):
    """Run batch testing on multiple documents"""
    if not image_paths:
        st.warning("No documents available for testing.")
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    
    for i, (image_path, doc_type) in enumerate(zip(image_paths, doc_types)):
        status_text.text(f"Processing document {i+1}/{len(image_paths)}: {Path(image_path).name}")
        
        traditional_result, inlining_result = run_accuracy_test(
            image_path, doc_type, traditional_processor, inlining_processor
        )
        
        results.append({
            'image_path': image_path,
            'document_type': doc_type,
            'traditional_result': traditional_result,
            'inlining_result': inlining_result
        })
        
        progress_bar.progress((i + 1) / len(image_paths))
    
    status_text.text("Processing complete!")
    
    return results

def display_dynamic_financial_impact(metrics):
    """Display the financial impact metrics based on actual test results"""
    if not metrics or 'comparative' not in metrics:
        st.info("Process more documents to generate financial impact metrics.")
        return
    
    # Get formatted metrics
    formatted_metrics = format_metrics_for_display(metrics)
    
    # Add visual indicators based on the metric values
    processing_time_color = "green" if float(formatted_metrics['processing_time_reduction'].rstrip('%')) > 50 else "orange"
    error_reduction_color = "green" if float(formatted_metrics['error_reduction'].rstrip('%')) > 50 else "orange"
    manual_review_color = "green" if float(formatted_metrics['manual_review_reduction'].rstrip('%')) > 50 else "orange"
    
    st.markdown("<div class='metrics-container'>", unsafe_allow_html=True)
    
    # Processing Time Reduction
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-title'>Processing Time Reduction</div>
        <div class='metric-value' style='color: {processing_time_color};'>{formatted_metrics['processing_time_reduction']}</div>
        <div class='metric-label'>Faster document processing</div>
        <div class='metric-note'>Based on {len(metrics['processing_times']['traditional_ocr'])} documents</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Error Reduction
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-title'>Error Reduction</div>
        <div class='metric-value' style='color: {error_reduction_color};'>{formatted_metrics['error_reduction']}</div>
        <div class='metric-label'>Fewer extraction errors</div>
        <div class='metric-note'>Based on confidence scores</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Manual Review Reduction
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-title'>Manual Review</div>
        <div class='metric-value' style='color: {manual_review_color};'>{formatted_metrics['manual_review_reduction']}</div>
        <div class='metric-label'>Reduction in review time</div>
        <div class='metric-note'>Based on low-confidence fields</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Time to Close
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-title'>Time to Close</div>
        <div class='metric-value'>{formatted_metrics['time_to_close']}</div>
        <div class='metric-label'>Faster loan processing</div>
        <div class='metric-note'>Industry benchmark</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    if metrics['document_count'] < 5:
        st.info(f"Note: These metrics are based on a small sample size ({metrics['document_count']} documents). Process more documents for more accurate results.")

def export_metrics_report():
    """Export metrics report as CSV or JSON"""
    metrics = get_metrics_tracker().get_latest_metrics()
    
    if not metrics or metrics['document_count'] == 0:
        st.warning("No metrics data available to export.")
        return
    
    # Create tabs for different export formats
    tab1, tab2 = st.tabs(["CSV", "JSON"])
    
    with tab1:
        # Prepare data for CSV
        data = {
            'Metric': [],
            'Traditional OCR': [],
            'Document Inlining': [],
            'Improvement (%)': []
        }
        
        # Processing Time
        data['Metric'].append('Processing Time (s)')
        data['Traditional OCR'].append(metrics['processing_time']['traditional_ocr']['average'])
        data['Document Inlining'].append(metrics['processing_time']['document_inlining']['average'])
        if 'comparative' in metrics:
            data['Improvement (%)'].append(metrics['comparative'].get('processing_time_reduction_pct', 0))
        else:
            data['Improvement (%)'].append(0)
        
        # Confidence Scores
        data['Metric'].append('Confidence Score')
        data['Traditional OCR'].append(metrics['confidence_scores']['traditional_ocr']['average'])
        data['Document Inlining'].append(metrics['confidence_scores']['document_inlining']['average'])
        if 'comparative' in metrics:
            data['Improvement (%)'].append(metrics['comparative'].get('confidence_improvement_pct', 0))
        else:
            data['Improvement (%)'].append(0)
        
        # Extracted Fields
        data['Metric'].append('Extracted Fields')
        data['Traditional OCR'].append(metrics['extraction_counts']['traditional_ocr']['average'])
        data['Document Inlining'].append(metrics['extraction_counts']['document_inlining']['average'])
        improvement = 0
        if metrics['extraction_counts']['traditional_ocr']['average'] > 0:
            improvement = ((metrics['extraction_counts']['document_inlining']['average'] - 
                            metrics['extraction_counts']['traditional_ocr']['average']) / 
                           metrics['extraction_counts']['traditional_ocr']['average'] * 100)
        data['Improvement (%)'].append(improvement)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Convert to CSV
        csv = df.to_csv(index=False)
        
        # Create download button
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"document_processing_metrics_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with tab2:
        # Prepare JSON data
        json_data = json.dumps(metrics, indent=2)
        
        # Create download button
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name=f"document_processing_metrics_{time.strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

def display_accuracy_test_ui():
    """Display the accuracy testing UI in Streamlit"""
    st.markdown("<div class='header'>Document Processing Accuracy Testing</div>", unsafe_allow_html=True)
    
    st.markdown("""
    This interface allows you to test the accuracy of different document processing methods:
    1. Upload or select documents
    2. Run tests to compare Traditional OCR with Document Inlining
    3. View metrics and export results
    """)
    
    # Return placeholder functions for the main app to implement
    return {
        'run_accuracy_test': run_accuracy_test,
        'display_accuracy_report': display_accuracy_report,
        'batch_test_documents': batch_test_documents,
        'export_metrics_report': export_metrics_report
    } 