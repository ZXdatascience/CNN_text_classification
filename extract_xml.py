import xml.etree.ElementTree as ET
import re

def extract(i):
    tree= ET.parse('/resources/data/XML/%s.xml' %i)
    meta_data= tree.findall("./head/meta")
    meta= {data.get("name"): data.get("content") for data in meta_data}
    general_desc= tree.findall(""".//identified-content/classifier[@type='general_descriptor']""")
    general_description= {general_desc[0].get('type'): [data.text for data in general_desc]}
    words= tree.findall("./head/pubdata")
    word_count= {"word_count": word.get("item-length") for word in words}
    headline= tree.findall("./body//hedline/hl1")
    title= {"headline": hl.text for hl in headline}
    byLine = tree.findall("./body//byline")
    byline = {"byline": [line.text for line in byLine]}
    author_ = tree.findall("./body//byline[@class='normalized_byline']")
    author = {"autor": auth.text for auth in author_}
    lead_para= tree.findall("./body//*block[@class='lead_paragraph']/p")
    lead_paragraph= {"lead_paragraph": "\n".join([data.text for data in lead_para])}
    ft= tree.findall("./body//*block[@class='full_text']/p")
    full_text= {"full_text": "\n".join([re.sub('(\n)( )+',' ',data.text) for data in ft])}
    result= {}
    for dictionary in [meta, general_description, word_count, title, byline, author, lead_paragraph, full_text]:
        result.update(dictionary)
    return result
