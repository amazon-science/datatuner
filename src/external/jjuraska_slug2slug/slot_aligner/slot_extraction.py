import os
import json
from nltk.tokenize import word_tokenize

from external.jjuraska_slug2slug import config


def extract_city(user_input, input_tokens, named_entities):
    city = None

    for ne in named_entities:
        if ne[0] == 'City':
            city = ne[2]
            break

    return city


def extract_eat_type(user_input):
    bar_synonyms = ['bar', 'bistro', 'brasserie', 'inn', 'tavern']
    coffee_shop_synonyms = ['café', 'cafe', 'coffee shop', 'coffeehouse', 'teahouse']
    restaurant_synonyms = ['cafeteria', 'canteen', 'chophouse', 'coffee shop', 'diner', 'donut shop', 'drive-in',
                           'eatery', 'eating place', 'fast-food place', 'joint', 'pizzeria', 'place to eat',
                           'restaurant', 'steakhouse']

    if any(x in user_input for x in bar_synonyms):
        return 'bar'
    elif any(x in user_input for x in coffee_shop_synonyms):
        return 'coffee shop'
    elif any(x in user_input for x in restaurant_synonyms):
        return 'restaurant'
    else:
        return None


def extract_categories(user_input, input_tokens):
    # file_categories_restaurants = 'dialogue/dialogue_modules/slug2slug/data/yelp/categories_restaurants.json'
    file_categories_restaurants = os.path.join(config.DATA_DIR, 'yelp', 'categories_restaurants.json')

    with open(file_categories_restaurants, 'r') as f_categories:
        categories = json.load(f_categories)

        for i, token in enumerate(input_tokens):
            # search for single-word occurrences in the category list
            if token in categories:
                return {'title': token,
                        'ids': categories[token]}

            # search for bigram occurrences in the category list
            if i > 0:
                key = ' '.join(input_tokens[i-1:i+1])
                if key in categories:
                    return {'title': key,
                            'ids': categories[key]}

    return {'title': None,
            'ids': []}


def extract_price_range(user_input, input_tokens):
    CHEAP = ['1', '2']
    MODERATE = ['2', '3']
    HIGH = ['3', '4']

    indicators_indep = {'cheap': CHEAP,
                        'inexpensive': CHEAP,
                        'affordable': CHEAP,
                        'modest': CHEAP,
                        'budget': CHEAP,
                        'economic': CHEAP,
                        'economical': CHEAP,
                        'expensive': HIGH,
                        'costly': HIGH,
                        'fancy': HIGH,
                        'posh': HIGH,
                        'stylish': HIGH,
                        'elegant': HIGH,
                        'extravagant': HIGH,
                        'luxury': HIGH,
                        'luxurious': HIGH}

    indicators_indep_bigram = {'low cost': CHEAP,
                               'high class': HIGH}

    indicators_priced = {'low': CHEAP,
                         'reasonably': CHEAP,
                         'moderately': MODERATE,
                         'high': HIGH,
                         'highly': HIGH}

    indicators_range = {'low': CHEAP,
                        'moderate': MODERATE,
                        'average': MODERATE,
                        'ordinary': MODERATE,
                        'middle': MODERATE,
                        'high': HIGH}

    # search for single-word occurrences in the indicator list
    for token in input_tokens:
        if token in indicators_indep:
            return indicators_indep[token]

    # search for bigram occurrences in the category list
    for key, val in indicators_indep_bigram.items():
        if key in user_input:
            return val

    idx = -1
    try:
        idx = input_tokens.index('priced')
        if idx > 0:
            prev_token = input_tokens[idx - 1]
            if prev_token in indicators_priced:
                return indicators_priced[prev_token]
    except ValueError:
        try:
            idx = input_tokens.index('price')
        except ValueError:
            try:
                idx = input_tokens.index('prices')
            except ValueError:
                pass

        if idx > 0:
            prev_token = input_tokens[idx - 1]
            if prev_token in indicators_range:
                return indicators_range[prev_token]

    return None


def extract_area(user_input, input_tokens):
    indicators_area = ['downtown', 'city center', 'city centre', 'center of', 'centre of', 'middle of']

    area = None

    for ind in indicators_area:
        if ind in user_input:
            area = 'downtown'
            break

    return area


def extract_family_friendly(user_input, input_tokens):
    indicators = ['family', 'families', 'child', 'children', 'kid', 'kids']

    for ind in indicators:
        if ind in user_input:
            return True

    return False


# TODO: implement
def extract_near(user_input):
    indicators = ['near', 'near to', 'close to', 'next to', 'neighborhood of', 'vicinity of']

    return None


def identify_slots(user_input, named_entities):
    attributes = {}

    user_input = user_input.lower()
    input_tokens = word_tokenize(user_input)

    city = extract_city(user_input, input_tokens, named_entities)
    if city:
        attributes['city'] = city

    eat_type = extract_eat_type(user_input)
    if eat_type:
        attributes['eatType'] = eat_type

    categories = extract_categories(user_input, input_tokens)
    if categories:
        attributes['categories'] = categories

    prices = extract_price_range(user_input, input_tokens)
    if prices:
        attributes['prices'] = prices

    family_friendly = extract_family_friendly(user_input, input_tokens)
    if family_friendly:
        attributes['familyFriendly'] = family_friendly

    area = extract_area(user_input, input_tokens)
    if area:
        attributes['area'] = area

    return attributes


# ---- MAIN ----

def main():
    user_input = 'Is there a family-friendly bar in downtown santa cruz that serves reasonably priced burgers?'
    gnode_entities = [('VisualArtwork', 282.797767, 'restaurant in'), ('City', 2522.766114, 'Santa Cruz')]
    print(identify_slots(user_input, gnode_entities))


if __name__ == '__main__':
    main()
