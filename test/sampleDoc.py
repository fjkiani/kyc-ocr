import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch

def create_sample_loan_application_pdf(output_path="sample_loan_application.pdf"):
    # Create a canvas to draw on
    c = canvas.Canvas(output_path, pagesize=letter)

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(1 * inch, 10.0 * inch, "Loan Application Form")

    # Section 1: Borrower Information
    c.setFont("Helvetica-Bold", 12)
    c.drawString(0.75 * inch, 9.5 * inch, "1. Borrower Information")

    c.setFont("Helvetica", 12)
    c.drawString(1.0 * inch, 9.2 * inch, "Full Name: John Doe")
    c.drawString(1.0 * inch, 9.0 * inch, "Address: 1234 Elm Street, City, ST 12345")
    c.drawString(1.0 * inch, 8.8 * inch, "Phone: (123) 456-7890")
    c.drawString(1.0 * inch, 8.6 * inch, "Email: johndoe@example.com")
    c.drawString(1.0 * inch, 8.4 * inch, "Social Security Number: 123-45-6789")

    # Section 2: Employment Information
    c.setFont("Helvetica-Bold", 12)
    c.drawString(0.75 * inch, 8.0 * inch, "2. Employment Information")

    c.setFont("Helvetica", 12)
    c.drawString(1.0 * inch, 7.7 * inch, "Employer: ABC Corporation")
    c.drawString(1.0 * inch, 7.5 * inch, "Position: Software Engineer")
    c.drawString(1.0 * inch, 7.3 * inch, "Monthly Income: $5,000")
    c.drawString(1.0 * inch, 7.1 * inch, "Years at Current Job: 3")

    # Section 3: Loan Information
    c.setFont("Helvetica-Bold", 12)
    c.drawString(0.75 * inch, 6.7 * inch, "3. Loan Information")

    c.setFont("Helvetica", 12)
    c.drawString(1.0 * inch, 6.4 * inch, "Loan Amount Requested: $20,000")
    c.drawString(1.0 * inch, 6.2 * inch, "Loan Purpose: Home Renovation")
    c.drawString(1.0 * inch, 6.0 * inch, "Collateral: None")

    # Section 4: References / Additional Information
    c.setFont("Helvetica-Bold", 12)
    c.drawString(0.75 * inch, 5.6 * inch, "4. References / Additional Information")

    c.setFont("Helvetica", 12)
    c.drawString(1.0 * inch, 5.3 * inch, "Additional Comments: Looking to repay over 5 years.")
    c.drawString(1.0 * inch, 5.1 * inch, "Reference 1: Jane Smith, (987) 654-3210")
    c.drawString(1.0 * inch, 4.9 * inch, "Reference 2: Bob Johnson, (555) 123-4567")

    # Signature
    c.drawString(0.75 * inch, 4.5 * inch, "Signature: John Doe")
    c.drawString(0.75 * inch, 4.3 * inch, "Date: 04/02/2025")

    # Save the PDF
    c.showPage()
    c.save()
    print(f"PDF generated at: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    create_sample_loan_application_pdf()
