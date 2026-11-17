import ast,json,sys,glob,difflib,os
P=os.path.expanduser('~/mnt/Volumes/Extreme_SSD/Javier_Santiago_Gaston_de_Iriarte_canrera/tools/book_study/')
R=os.path.expanduser('~/mnt/GitHub/Adv_Fin_ML_Exercises/')
def funcs_from_src(src,tag):
    out={}
    try: t=ast.parse(src)
    except SyntaxError as e:
        # try cell-by-cell fallback handled by caller
        raise
    for n in t.body:
        if isinstance(n,(ast.FunctionDef,ast.ClassDef)):
            # strip docstrings
            for x in ast.walk(n):
                if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.body and isinstance(x.body[0],ast.Expr) and isinstance(getattr(x.body[0],'value',None),ast.Constant) and isinstance(x.body[0].value.value,str):
                    x.body=x.body[1:] or [ast.Pass()]
            out.setdefault(n.name,[]).append((tag,ast.unparse(n)))
    return out
def merge(a,b):
    for k,v in b.items(): a.setdefault(k,[]).extend(v)
mine={}
for f in ['book_port_a.py','book_port_b.py','book_port_e.py','book_mp.py']:
    merge(mine,funcs_from_src(open(P+f).read(),f))
theirs={}
merge(theirs,funcs_from_src(open(R+'src/features/snippets.py').read(),'snippets.py'))
for nb in glob.glob(R+'notebooks/*.ipynb'):
    for i,c in enumerate(json.load(open(nb))['cells']):
        if c['cell_type']!='code': continue
        s='\n'.join(l for l in ''.join(c['source']).splitlines() if not l.lstrip().startswith(('%','!')))
        try: merge(theirs,funcs_from_src(s,os.path.basename(nb)+f'#cell{i}'))
        except SyntaxError: pass
names=sys.argv[1:] or sorted(set(mine)&set(theirs))
for k in names:
    if k not in mine or k not in theirs: print(f'{k}: missing in', 'mine' if k not in mine else 'theirs'); continue
    best=None
    for tm,sm in mine[k]:
        for tt,st in theirs[k]:
            d=list(difflib.unified_diff(sm.splitlines(),st.splitlines(),f'mine:{tm}',f'theirs:{tt}',lineterm='',n=0))
            if best is None or len(d)<len(best[0]): best=(d,tm,tt)
    d,tm,tt=best
    print(f'### {k}: {"IDENTICAL" if not d else "DIFFERS"}  ({tm} vs {tt})')
    if d and '--full' in os.environ.get('MODE',''): print('\n'.join(d))
