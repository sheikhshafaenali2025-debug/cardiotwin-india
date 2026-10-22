"""Export the CardioTwin pipeline as a one-page PDF with ReportLab."""
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.pdfgen import canvas

OUT=Path(__file__).with_name("architecture.pdf")
W,H=1000,562
c=canvas.Canvas(str(OUT),pagesize=(W,H))
c.setTitle("CardioTwin India - System Architecture")
c.setFillColor(colors.HexColor("#F7F8F5")); c.rect(0,0,W,H,fill=1,stroke=0)
c.setFillColor(colors.HexColor("#153A38")); c.setFont("Helvetica-Bold",20); c.drawString(42,H-46,"CardioTwin India | System architecture")
c.setFillColor(colors.HexColor("#526A68")); c.setFont("Helvetica",9); c.drawString(42,H-64,"Synthetic-data research prototype · heuristic proxy label · not for clinical use")
nodes=[
 (40,386,155,66,"Synthetic EHR","Demographics, conditions,\nlabs, BP and medication"),
 (40,258,155,66,"Synthetic wearable","Configurable hourly HR, HRV,\nsleep, steps and BP"),
 (245,320,170,82,"Input validation","Schema, units, India time zone\nand completeness"),
 (465,320,165,82,"Patient state","History and rolling\npersonal baseline"),
 (665,320,165,82,"Causal features","Daily summaries and\nmissingness flags"),
 (665,183,165,74,"Model comparison","Logistic regression, XGBoost\nand CNN-GRU"),
 (850,305,120,100,"Clinician UI","Synthetic proxy score,\nreason signals and\ninput scenarios"),
 (245,120,210,72,"Proxy-label generator","Transparent next-window rule;\nnot a clinical outcome"),
 (500,120,255,72,"Evaluation","Patient-grouped train, validation\nand holdout; synthetic labels only"),
]
def arrow(a,b):
    x1,y1=a; x2,y2=b; c.setStrokeColor(colors.HexColor("#4E7772")); c.setLineWidth(1.5); c.line(x1,y1,x2,y2)
    import math
    angle=math.atan2(y2-y1,x2-x1); size=6
    for off in (0.5,-0.5): c.line(x2,y2,x2-size*math.cos(angle+off),y2-size*math.sin(angle+off))
def node(x,y,w,h,title,body):
    c.setFillColor(colors.white); c.setStrokeColor(colors.HexColor("#8BA9A4")); c.roundRect(x,y,w,h,8,fill=1,stroke=1)
    c.setFillColor(colors.HexColor("#176B63")); c.setFont("Helvetica-Bold",10); c.drawCentredString(x+w/2,y+h-22,title)
    c.setFillColor(colors.HexColor("#263E3D")); c.setFont("Helvetica",8)
    for i,line in enumerate(body.split("\n")): c.drawCentredString(x+w/2,y+h-40-i*12,line)
for n in nodes: node(*n)
arrow((195,419),(245,375)); arrow((195,291),(245,345)); arrow((415,361),(465,361)); arrow((630,361),(665,361)); arrow((747,320),(747,257)); arrow((830,220),(850,330)); arrow((195,419),(245,156)); arrow((195,291),(245,156)); arrow((455,156),(500,156))
# Evaluation informs model development, while each patient's windows remain in one split.
c.setStrokeColor(colors.HexColor("#B88036")); c.setLineWidth(1.2); c.setDash(4,3); c.line(627,192,627,220); arrow((627,220),(665,220)); c.setDash()
c.setFillColor(colors.HexColor("#6D501D")); c.setFont("Helvetica-Bold",8); c.drawCentredString(630,102,"Patient-grouped splits keep windows from one person together")
c.setFillColor(colors.HexColor("#526A68")); c.setFont("Helvetica",8); c.drawString(42,28,"Synthetic proxy outcomes only. Scores are not calibrated clinical event probabilities or medical advice.")
c.save(); print(OUT)
