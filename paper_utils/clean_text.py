str_list = \
"""APPNP 77.55 ± 0.05 n.a. n.a. 54.57 ± 0.14 n.a. n.a.
VGCN 77.89 ± 0.05 n.a. n.a. 54.87 ± 0.12 n.a. n.a.
VGCN-Dropout 78.11 ± 0.05 71.74 ± 0.14 n.a. 55.40 ± 0.13 43.43 ± 0.18 n.a.
VGCN-Energy 77.89 ± 0.05 n.a. n.a. 54.87 ± 0.12 n.a. n.a.
VGCN-Ensemble 78.14 71.48 n.a. 53.95 42.87 n.a.
GKDE-GCN 77.47 ± 0.33 77.55 ± 0.33 n.a. 61.62 ± 1.00 62.33 ± 1.00 n.a.
GPN 75.44 ± 0.19 72.71 ± 0.28 61.45 ± 0.49 55.64 ± 0.37 52.99 ± 0.49 39.37 ± 0.42"""
keep_model = [
    'APPNP', 'VGCN-Dropout', 'GKDE-GCN', "GPN"
]
str_list = str_list.split("\n")
for a_str in str_list:
    strs = a_str.split(" ")
    if not strs[0] in keep_model:
        continue
    out = []
    use_next = True
    for a_str in strs:
        if a_str == "±":
            use_next = False
            continue 
        
        if use_next:
            out.append(a_str)
        else:
            use_next=True
    print("\t ".join(out))