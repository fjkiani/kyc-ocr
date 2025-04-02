# Speaker Notes - Technical Discovery Meeting

## Title Slide

**What to say:**
"Good [morning/afternoon], everyone. I'm [Your Name], a Solutions Architect at Fireworks AI. Thank you for taking the time to meet with me today to discuss how we might help streamline your loan processing systems. I'm looking forward to learning more about your specific challenges and goals."

**Body language:**
- Make eye contact with each person in the room
- Project confidence and expertise
- Smile and appear approachable

## Agenda Slide

**What to say:**
"Here's how I've structured our time together today. We'll start with brief introductions, then I'd like to spend the majority of our time understanding your current process and the challenges you're facing. Based on what I learn, I'll share some relevant capabilities from Fireworks AI that might address those challenges. We'll finish with time for questions and discussion about potential next steps.

The key to a successful meeting today is for me to gain a deep understanding of your specific needs before suggesting potential solutions."

**Note to self:**
- This sets expectations that this is a discovery meeting, not a sales pitch
- Emphasize that you're here to listen first, then offer targeted insights

## Introduction Slide

**What to say:**
"Before we dive into your specific situation, I'd like to share a brief background on Fireworks AI. We specialize in document intelligence using our Compound AI approach, which bridges the gap between different data formats. Our Document Inlining technology transforms images and PDFs into formats that allow language models to better process and reason about their contents.

In our recent evaluations, our approach outperformed GPT-4o on document processing tasks in 68% of test cases. This is particularly relevant for complex financial documents like those used in loan processing."

**Tactical note:**
- Keep this very brief - no more than 1 minute
- Focus on the technology approach rather than client numbers
- The goal is to establish technical credibility, not to oversell

## About You Slide

**What to say:**
"But enough about us - I'm much more interested in learning about you and your team. Could you each briefly introduce yourselves and share your role in the loan processing initiative? I'm also curious about what prompted you to explore AI solutions at this time, and if you have a particular timeline in mind for implementation."

**How to handle responses:**
- Take brief notes on each person's role and motivation
- Note who seems most engaged or concerned
- Listen for mentions of specific pain points or goals
- Pay attention to any mentioned deadlines or timelines

## Understanding Your Current Process Slide

**What to say:**
"Now I'd like to understand your current loan processing workflow. Could you walk me through the typical journey of a loan application from initial submission to closing? I'm particularly interested in understanding which parts of the process are manual versus automated, your current volume, average time-to-close, and how many different systems are involved."

**Follow-up questions if needed:**
- "What happens after the application is received?"
- "How is the data extracted from documents?"
- "Who is responsible for verification steps?"
- "What triggers the underwriting process?"
- "How are regulatory requirements addressed?"

**Note-taking focus:**
- Document the major steps in their workflow
- Identify manual touchpoints
- Note systems mentioned
- Mark areas where they express frustration

## Common Pain Points Slide

**What to say:**
"Based on our work with other financial institutions, these are some of the common pain points in loan processing. These include challenges with document processing and data extraction, keeping up with regulatory compliance, the burden of manual review and verification, process bottlenecks, and integration challenges between systems.

Which of these resonate most with your experience? Or perhaps you're facing different challenges entirely?"

**How to handle responses:**
- Let them talk - this is where you'll get valuable insights
- Note which pain points they emphasize
- Ask clarifying questions about severity and impact
- Listen for organizational priorities

## Your Specific Challenges Slide

**What to say:**
"Let's dig deeper into your specific challenges. Where do you see the most significant delays or bottlenecks in your current process? What aspects of document processing are most error-prone? How do compliance requirements impact your processing speeds? And importantly, what are your customers' biggest frustrations with the current process?"

**Note-taking focus:**
- Capture specific metrics if mentioned
- Note emotional emphasis on particular issues
- Connect challenges to business impact
- Identify which challenges might be most easily addressed

## Your Goals and Success Metrics Slide

**What to say:**
"Now that I understand your challenges better, I'd like to learn about your goals. What specific improvements are you hoping to achieve by streamlining your loan processing? How do you currently measure efficiency? What would success look like for this initiative in 6-12 months? Are there specific competitors or industry benchmarks you're measuring yourself against?"

**Why this matters:**
- Their answers will help you frame your capabilities in terms of their goals
- Success metrics are crucial for building an ROI case
- Understanding timeline expectations helps with next steps

## Fireworks AI Platform Overview Slide

**What to say:**
"Based on what you've shared about your challenges and goals, I'd like to highlight some relevant capabilities from our platform. Fireworks AI offers a comprehensive document intelligence platform that enables end-to-end automation of loan processing workflows. We integrate with existing loan origination systems and offer flexible deployment options including cloud, on-premises, or hybrid approaches.

The capabilities I'm about to share seem most relevant to the specific challenges you mentioned about [reference 2-3 key challenges they emphasized]."

**Delivery approach:**
- Tie each capability back to a specific challenge they mentioned
- Keep it high-level, focusing on outcomes rather than features
- Speak in terms of their business, not your technology

## Intelligent Document Processing Slide

**What to say:**
"Let's talk about the core technology that makes our approach unique. We call it 'Bridging the Document-AI Modality Gap' - and it's the fundamental reason our solution delivers superior results.

The challenge with AI processing of complex documents is what we call the 'modality gap.' Traditional approaches treat documents as images and use vision models to analyze them. But this approach loses critical information - especially with financial documents that have complex structures like tables, forms, and multi-page relationships.

Our Document Inlining technology takes a fundamentally different approach. Instead of treating documents as images, we transform them into formats that allow language models to process them with their full reasoning capabilities intact. 

Our Compound AI technology preserves the structure and relationships between document elements throughout this transformation. When a form becomes text, we maintain the knowledge that certain text was in a specific field, or that numbers in a table have relationships with row and column headers.

We then apply specialized language models that understand financial documents in depth. These models can reason about the content in ways that vision models simply cannot. For example, they can understand that a specific field represents a debt-to-income ratio and can evaluate whether that ratio aligns with other financial information in the application.

In our performance testing, this approach outperformed GPT-4o on 68% of document processing tasks. This isn't just an incremental improvement - it's a fundamentally better way to handle complex financial documents.

For your loan processing workflow, this means dramatically higher accuracy in extracted information, fewer manual verification steps, and faster processing times."

**Questions to ask:**
- "What are the most complex document types you deal with in your loan processing?"
- "How much time do your teams spend verifying AI-extracted information?"
- "What would a 15-20% improvement in extraction accuracy mean for your downstream processes?"

---

## Document Processing Architecture Slide

**What to say:**
"Let me walk you through this architecture diagram, which illustrates what makes our approach fundamentally different from traditional document AI solutions.

On the left, we have various document formats that are common in loan processing - PDFs, mobile images, scanned documents, and complex tables. Financial institutions deal with hundreds of these document types daily.

The key challenge in this space is what we call the 'modality gap' - shown in the middle of the diagram. Traditional AI vision models struggle with complex document structures. They can 'see' the documents but often lose critical information about structure, relationships between elements, and context.

Our unique solution is Fireworks AI's Document Inlining technology, which actually bridges this modality gap. Rather than treating documents as images, we transform them into a format that preserves their structure while making them processable by language models. This is fundamentally different from vision-only approaches.

This transformation feeds into our Compound AI technology, which maintains structural relationships throughout processing. For instance, it recognizes that a table on a bank statement should remain a table, with relationships between columns and rows preserved, not just a collection of text snippets.

Finally, we apply specialized language models with deep financial domain knowledge to reason about the documents. These models understand the significance of specific fields in financial contexts - they know what a debt-to-income ratio means and can identify inconsistencies across documents.

The result is dramatically better business outcomes, as shown on the right: 85% faster processing, over 95% extraction accuracy, reduced operating costs, and an improved customer experience with faster loan closings.

What makes this approach powerful is that we're not just extracting text - we're preserving context, maintaining relationships, and enabling sophisticated reasoning that simply isn't possible with traditional document AI approaches."

**Questions to engage with:**
- "How does your current document processing solution handle complex financial tables and forms?"
- "What kind of accuracy rates are you seeing with your current document extraction methods?"
- "How much time is your team spending manually reviewing and correcting AI-extracted data?"
- "What would it mean for your operations if you could reduce document processing time by 85%?"

**Note to self:**
- Listen carefully for pain points around manual verification and correction
- Note any comments about complexity of their document types
- Pay attention to current accuracy rates they mention - our 95%+ accuracy is often a key differentiator
- Watch for reactions when you mention the modality gap - many prospects haven't heard this concept before

---

## Key Capabilities Slide

**What to say:**
"Building on the document processing architecture we just discussed, let me highlight three key capability areas that would directly address the challenges you've mentioned:

First, our workflow automation capabilities. Based on the bottlenecks you described in your process, we can automate document classification and routing, verify information against multiple data sources in real-time, and handle exceptions with intelligent human oversight when needed. Our clients typically see around 70% reduction in manual processing steps with these capabilities.

Second, compliance and audit capabilities. You mentioned the challenges with [reference their specific compliance pain points]. Our system provides automated regulatory checks with complete audit trails, tracks data provenance throughout the process, and ensures fair lending consistency. This typically reduces compliance review time by about 85%, while also improving accuracy and reducing regulatory risk.

Finally, our integration approach. We use an OpenAI-compatible API that requires just a one-line code change to enable document transformation. The system supports all document formats you're working with, offers flexible deployment options to match your IT requirements, and can be implemented in 8-12 weeks - significantly faster than the industry average of 6+ months.

What's important is that these capabilities work together as a unified solution that addresses the specific challenges financial institutions face with loan processing, focusing on reducing manual effort while improving both accuracy and compliance."

**Questions to ask:**
- "Which of these capabilities would have the biggest impact on your current process?"
- "How would reducing manual steps by 70% affect your team's capacity and loan throughput?"
- "What would be the value of having consistency verification for compliance purposes?"
- "How important is implementation timeline in your evaluation process?"

**Note to self:**
- Tailor emphasis based on their earlier responses about pain points
- Listen for which capability area generates the most interest
- Note any specific metrics they react to for the ROI conversation
- Pay attention to technical questions that might indicate implementation concerns

---

## Implementation Timeline Slide

**What to say:**
"Based on implementations with similar organizations, we typically complete a full implementation in 8-12 weeks. This breaks down into three main phases: document processing automation in 4-6 weeks, workflow integration in 2-3 weeks, and compliance automation in another 2-3 weeks.

This timeline is significantly faster than the industry average of 6+ months, allowing you to realize value much more quickly."

**If they ask about resources required:**
- Be prepared to discuss typical implementation team composition
- Mention level of effort required from their team
- Highlight project management approach

## Proven Results Slide

**What to say:**
"When evaluating our Document Inlining technology, we conducted tests comparing it against leading vision models like GPT-4o. In a rigorous evaluation of 100 document-question pairs using arXiv articles, our approach was preferred in 68% of cases when evaluated by Claude 3.5-Sonnet.

The key advantage is that we can leverage the superior reasoning capabilities of specialized language models rather than relying solely on vision models that often struggle with complex document structures. This means more accurate extraction of information, better understanding of relationships between different data points, and ultimately more reliable processing.

For loan processing, this translates to improved accuracy in document handling, reduced need for manual review, and more consistent extraction of key information from financial documents."

**Credibility enhancers:**
- Reference the specific testing methodology if asked for details
- Explain the technical reasons why this approach performs better
- Focus on the quality improvements rather than making specific ROI claims

## Discussion Slide

**What to say:**
"I've shared a brief overview of capabilities that seem most relevant to your challenges. Now I'd like to open up for discussion. Which of these capabilities seem most relevant to your needs? Are there specific areas you'd like to explore further? Are there other stakeholders who should be involved in future discussions? What would be most helpful for our next conversation?"

**Handling questions:**
- Listen carefully and don't interrupt
- Answer directly and honestly
- If you don't know, say so and offer to follow up
- Note topics of greatest interest for follow-up

## Recommended Next Steps Slide

**What to say:**
"Based on our discussion today, here are some potential next steps to consider. We could arrange a technical demonstration focused specifically on [reference their key challenges], conduct a solution workshop with a broader group of stakeholders, prepare an ROI analysis based on your loan volume and current metrics, or set up a proof of concept using your actual documents (anonymized, of course).

What makes the most sense as a next step for your team?"

**Closing the meeting effectively:**
- Get specific commitments if possible
- Suggest a concrete timeframe for the next step
- Identify who should be involved in follow-up

## Thank You Slide

**What to say:**
"Thank you for your time today. I've learned a great deal about your challenges and goals, and I'm excited about the potential to help streamline your loan processing operations. Here's my contact information for any follow-up questions that might arise. What's the best way to coordinate our next steps?"

**After the meeting:**
- Send follow-up email within 24 hours
- Include summary of key points discussed
- Attach relevant case studies based on their interests
- Confirm next steps and timeline 