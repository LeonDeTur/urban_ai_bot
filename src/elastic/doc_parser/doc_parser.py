import pandas as pd
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph


class DocParser:

    @staticmethod
    def iter_contexts_for_vectorization(parent: Document):

        for child in parent.element.body.iterchildren():
                if child.tag.endswith('}p'):
                    para = Paragraph(child, parent)
                    if para.text != '':
                        yield para.text, "text"
                elif child.tag.endswith('}tbl'):
                    table = Table(child, parent)
                    columns = []
                    col_counter = 0
                    for i in [cell.text for cell in table.rows[0].cells if cell]:
                        if i:
                            columns.append(i)
                        else:
                            columns.append(f'Без_названия_{col_counter}')
                            col_counter += 1

                    yield str(
                        pd.DataFrame(
                            [[cell.text for cell in row.cells] for row in table.rows[1:]],
                            columns=columns
                        ).to_dict('list')
                    ), "table"
