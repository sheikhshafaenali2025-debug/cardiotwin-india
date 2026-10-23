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
 (30,390,150,60,"Synthetic EHR","Demographics, conditions,\nlabs, BP and medication"),
 (30,275,150,60,"Synthetic wearable","Hourly HR, HRV, sleep,\nsteps and BP"),
 (220,330,145,70,"Input validation","Schema, units, India time zone\nand completeness"),
 (410,330,160,70,"Causal daily features","Daily summaries and\nmissingness flags"),
 (625,390,150,70,"Twin Drift","Prior 14-day baseline vs\nlatest 7 days; coverage-aware"),
 (625,275,150,70,"Model comparison","Logistic regression, XGBoost\nand CNN-GRU"),
 (820,330,150,70,"Research dashboard","Synthetic proxy score,\ntrend and data context"),
 (820,210,150,70,"Branching futures","14-day bootstrap paths;\nmodel response ranges"),
 (220,120,190,68,"Proxy-label generator","Transparent next-window rule;\nnot a clinical outcome"),
 (450,120,240,68,"Evaluation","Patient-grouped train, validation\nand holdout; synthetic labels only"),
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
arrow((180,420),(220,370)); arrow((180,305),(220,350)); arrow((365,365),(410,365)); arrow((570,365),(625,425)); arrow((570,350),(625,310)); arrow((775,310),(820,350)); arrow((775,310),(820,245)); arrow((895,280),(895,330)); arrow((180,420),(220,154)); arrow((180,305),(220,154)); arrow((410,154),(450,154))
# Evaluation informs model development, while each patient's windows remain in one split.
c.setStrokeColor(colors.HexColor("#B88036")); c.setLineWidth(1.2); c.setDash(4,3); c.line(690,188,690,235); arrow((690,235),(700,275)); c.setDash()
c.setFillColor(colors.HexColor("#6D501D")); c.setFont("Helvetica-Bold",8); c.drawCentredString(625,92,"Patient-grouped splits keep windows from one person together")
c.setFillColor(colors.HexColor("#526A68")); c.setFont("Helvetica",8); c.drawString(42,28,"Synthetic proxy outcomes only. Scores are not calibrated clinical event probabilities or medical advice.")
c.save(); print(OUT)
