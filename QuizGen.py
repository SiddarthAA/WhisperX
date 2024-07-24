from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        pass

    def title_page(self, title):
        self.add_page()
        self.set_font('Arial', '', 16)
        self.cell(0, 10, title, 0, 1, 'C')
        self.ln(15)

    def chapter_title(self, title):
        self.set_font('Arial', '', 12)
        self.cell(0, 10, title, 0, 1, 'C')
        self.ln(8)

    def chapter_body(self, body):
        self.set_font('Arial', '', 8)
        for line in body:
            self.multi_cell(0, 4, line)
        self.ln(2)

    def add_questions_section(self, mcqs, fill_in_the_blanks, true_false, short_answers):
        self.set_font('Arial', '', 8)

        self.chapter_title('Section A: Multiple Choice Questions')
        for idx, question in enumerate(mcqs):
            q_text = f"{idx + 1}. {question[0]}"
            self.multi_cell(0, 4, q_text)
            if len(question) > 1 and isinstance(question[1], list):
                for option in question[1]:
                    self.ln(2)
                    self.multi_cell(0, 4, f"   {option}")
            self.ln(7)

        self.ln(15)
        self.chapter_title('Section B: Fill in the Blanks')
        for idx, question in enumerate(fill_in_the_blanks):
            q_text = f"{idx + 1}. {question[0]}"
            self.multi_cell(0, 4, q_text)
            self.ln(4)

        self.ln(15)
        self.chapter_title('Section C: True or False')
        for idx, question in enumerate(true_false):
            q_text = f"{idx + 1}. {question[0]}"
            self.multi_cell(0, 4, q_text)
            self.ln(4)

        self.ln(15)
        self.chapter_title('Section D: Short Answer Questions')
        for idx, question in enumerate(short_answers):
            q_text = f"{idx + 1}. {question[0]}"
            self.multi_cell(0, 4, q_text)
            self.ln(4)

    def add_answers_section(self, mcqs, fill_in_the_blanks, true_false, short_answers):
        self.add_page()
        self.set_font('Arial', '', 8)

        self.chapter_title('Section A: Multiple Choice Questions - Answers')
        for idx, question in enumerate(mcqs):
            q_text = f"{idx + 1}. {question[0]}"
            a_text = f"Answer: {question[2]}" if len(question) > 2 else f"Answer: {question[1]}"
            self.multi_cell(0, 4, q_text)
            self.ln(1)
            self.multi_cell(0, 4, a_text)
            self.ln(8)

        self.ln(15)
        self.chapter_title('Section B: Fill in the Blanks - Answers')
        for idx, question in enumerate(fill_in_the_blanks):
            q_text = f"{idx + 1}. {question[0]}"
            a_text = f"Answer: {question[1]}"
            self.multi_cell(0, 4, q_text)
            self.ln(2)
            self.multi_cell(0, 4, a_text)
            self.ln(8)

        self.ln(15)
        self.chapter_title('Section C: True or False - Answers')
        for idx, question in enumerate(true_false):
            q_text = f"{idx + 1}. {question[0]}"
            a_text = f"Answer: {question[1]}"
            self.multi_cell(0, 4, q_text)
            self.ln(1)
            self.multi_cell(0, 4, a_text)
            self.ln(8)

        self.ln(15)
        self.chapter_title('Section D: Short Answer Questions - Answers')
        for idx, question in enumerate(short_answers):
            q_text = f"{idx + 1}. {question[0]}"
            a_text = f"Answer: {question[1]}"
            self.multi_cell(0, 4, q_text)
            self.ln(1)
            self.multi_cell(0, 4, a_text)
            self.ln(8)

def create_question_paper(mcqs, fill_in_the_blanks, true_false, short_answers, title='Question Paper', filename='QuestionBank.pdf'):
    pdf = PDF()
    pdf.title_page(title)
    pdf.add_questions_section(mcqs, fill_in_the_blanks, true_false, short_answers)
    pdf.add_answers_section(mcqs, fill_in_the_blanks, true_false, short_answers)
    pdf.output(filename)