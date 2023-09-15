python -m jupyter nbconvert --to python exploration.ipynb \
--TemplateExporter.exclude_input_prompt=True \
--TemplateExporter.exclude_output=True \
--TemplateExporter.exclude_output_prompt=True \
--TemplateExporter.exclude_markdown=True \
--TemplateExporter.exclude_raw=True\