from xml.etree import ElementTree


def camel_case_split(s):
    words = [[s[0]]]

    for c in s[1:]:
        if words[-1][-1].islower() and c.isupper():
            words.append(list(c))
        else:
            words[-1].append(c)

    return " ".join(["".join(word).lower() for word in words])


def cleanup(s):
    if type(s) != str:
        s = ElementTree.tostring(s, encoding="unicode")
    s = s.replace("\t", " ").replace("\n", " ").replace("_", " ")
    s = " ".join(s.split(" ")).strip()
    return s
