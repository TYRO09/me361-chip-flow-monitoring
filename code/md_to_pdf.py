"""Render a Markdown file to PDF with headless Chrome (images resolved relative to the .md)."""
import os, re, subprocess, sys
import markdown

CSS = """
@page{size:A4;margin:15mm 14mm}
body{font-family:-apple-system,"Helvetica Neue",Arial,sans-serif;font-size:10.5pt;line-height:1.45;color:#1a1a1a}
h1{font-size:19pt;margin:0 0 6px;border-bottom:3px solid #1f4e79;padding-bottom:6px;color:#1f4e79}
h2{font-size:13.5pt;color:#1f4e79;margin:18px 0 6px;break-after:avoid}
h3{font-size:11.5pt;margin:12px 0 4px;break-after:avoid}
table{border-collapse:collapse;width:100%;margin:6px 0;font-size:9.5pt;break-inside:avoid}
th,td{border:1px solid #bbb;padding:3px 6px;text-align:left;vertical-align:top}
th{background:#e8eef5}
blockquote{margin:6px 0;padding:6px 10px;background:#f3f6fa;border-left:4px solid #1f4e79}
code{background:#f1f1f1;padding:0 3px;border-radius:3px;font-size:9pt}
img{max-width:100%;display:block;margin:6px auto;break-inside:avoid}
hr{border:none;border-top:1px solid #ddd;margin:12px 0}
ul,ol{margin:4px 0;padding-left:20px} li{margin:2px 0}
"""
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def convert(md_path, pdf_path=None):
    md_path = os.path.abspath(md_path)
    pdf_path = pdf_path or md_path[:-3] + ".pdf"
    md = open(md_path).read()
    md = re.sub(r"(?m)^(?![ \t]*[-*] |[ \t]*\d+\. |\s*$)(.+)\n(?=[ \t]*(- |\d+\. ))", r"\1\n\n", md)
    md = re.sub(r"(?m)^ {2,3}- ", "    - ", md)  # Python-Markdown needs 4-space nesting
    md = md.replace("- [ ] ", "- ☐ ")
    body = markdown.markdown(md, extensions=["tables", "sane_lists"])
    html_path = md_path[:-3] + ".print.html"
    with open(html_path, "w") as fh:
        fh.write(f"<!doctype html><html><head><meta charset='utf-8'><base href='file://{os.path.dirname(md_path)}/'>"
                 f"<style>{CSS}</style></head><body>{body}</body></html>")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--allow-file-access-from-files",
                    f"--print-to-pdf={pdf_path}", f"file://{html_path}"], check=True, capture_output=True)
    os.remove(html_path)
    return pdf_path


if __name__ == "__main__":
    print(convert(*sys.argv[1:]))
