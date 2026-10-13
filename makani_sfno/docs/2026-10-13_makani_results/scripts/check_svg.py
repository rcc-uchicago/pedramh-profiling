"""Layout check without a renderer: estimated text boxes vs canvas, other text, and plotted lines."""
import re, sys, xml.dom.minidom as md
def boxes(path):
    d=md.parse(path); svg=d.documentElement; W,H=[float(v) for v in svg.getAttribute("viewBox").split()[2:]]
    T=[];L=[]
    for t in d.getElementsByTagName("text"):
        txt="".join(n.data for n in t.childNodes if n.nodeType==n.TEXT_NODE)
        cls=t.getAttribute("class"); fs={"t":14,"st":12,"tick":11,"lab":12}.get(cls,12); w=len(txt)*fs*(0.60 if cls=="t" else 0.56)
        x=float(t.getAttribute("x")); y=float(t.getAttribute("y")); a=t.getAttribute("text-anchor") or "start"
        x0={"start":x,"middle":x-w/2,"end":x-w}[a]; T.append((x0,y-fs*0.78,x0+w,y+fs*0.22,txt))
    for p in d.getElementsByTagName("path"):
        pts=[tuple(map(float,q.split(","))) for q in re.findall(r"[-\d.]+,[-\d.]+",p.getAttribute("d"))]
        L.append(pts)
    return W,H,T,L
def seg_hits(b,p,q,n=40):
    for i in range(n+1):
        x=p[0]+(q[0]-p[0])*i/n; y=p[1]+(q[1]-p[1])*i/n
        if b[0]-1<=x<=b[2]+1 and b[1]-1<=y<=b[3]+1: return True
    return False
for f in sys.argv[1:]:
    W,H,T,L=boxes(f); bad=0
    for b in T:
        if b[0]<2 or b[2]>W-2 or b[1]<0 or b[3]>H: print("  OUT  ",round(b[0]),round(b[2]),b[4][:60]); bad+=1
    for i in range(len(T)):
        for j in range(i+1,len(T)):
            a,b=T[i],T[j]
            if a[0]<b[2] and b[0]<a[2] and a[1]<b[3] and b[1]<a[3]: print("  TEXT ",a[4][:40],"|",b[4][:40]); bad+=1
    for b in T:
        for pts in L:
            if any(seg_hits(b,pts[k],pts[k+1]) for k in range(len(pts)-1)): print("  LINE ",b[4][:60]); bad+=1; break
    print(f.rsplit("/",1)[1], "issues:",bad)
