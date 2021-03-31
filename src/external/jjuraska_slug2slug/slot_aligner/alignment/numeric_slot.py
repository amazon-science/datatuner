from external.jjuraska_slug2slug.slot_aligner.alignment.utils import find_first_in_list


def align_numeric_slot_with_unit(text, text_tok, slot, value):
    value_number = value.split(' ')[0]
    try:
        float(value_number)
    except ValueError:
        return -1

    _, pos = find_first_in_list(value_number, text_tok)

    return pos


def align_year_slot(text, text_tok, slot, value):
    try:
        int(value)
    except ValueError:
        return -1

    year_alternatives = [value]
    if len(value) == 4:
        year_alternatives.append('\'' + value[-2:])
        year_alternatives.append(value[-2:])

    for val in year_alternatives:
        if len(val) > 2:
            pos = text.find(val)
        else:
            _, pos = find_first_in_list(val, text_tok)

        if pos >= 0:
            return pos

    return -1
