#  Converter functions available to apply to text fields

def clean_mrl(mrl):
    return mrl.replace("_", " ").replace(".", " ").replace("(", " ( ").replace(")", " ) ")


converters = {"clean_mrl": clean_mrl}
