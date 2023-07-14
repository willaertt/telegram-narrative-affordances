'''
NLP functions for extracting actants from texts
'''

#import libraries
import spacy
nlp = spacy.load('en_core_web_sm', disable = ['ner'])

#define helper functions
def get_pos_chunks(sent, verb, dep):
    '''
    takes sentence, verb, dependency tag; returns noun chunks per dependency tag
    '''
    sent_chunks = [chunk for chunk in sent.noun_chunks]

    # regular prepositional objects that depend on the verb
    preposition = None
    if dep in ["pobj", "agent"]:
        chunk_list = []
        base_list = [
            chunk for chunk in sent_chunks if chunk.root.head in verb.children and chunk.root.dep_ == dep]
        for chunk in base_list:
            preposition = chunk.root.head
            chunk_list.append(chunk)
            conjuncts = [conjunct for conjunct in chunk.root.conjuncts]
            for chunk2 in sent_chunks:
                if chunk2.root in conjuncts:
                    chunk_list.append(chunk2)

    # no relative clause
    elif verb.dep_ != "relcl":
        chunk_list = []
        base_list = [
            chunk for chunk in sent_chunks if chunk.root in verb.children and chunk.root.dep_ == dep]
        for chunk in base_list:
            chunk_list.append(chunk)
            conjuncts = [conjunct for conjunct in chunk.root.conjuncts]
            for chunk2 in sent_chunks:
                if chunk2.root in conjuncts:
                    chunk_list.append(chunk2)

    # relative clause
    elif verb.dep_ == "relcl":
        chunk_list = []
        base_list = [
            chunk for chunk in sent_chunks if chunk.root in verb.children and chunk.root.dep_ == dep]
        lefts = [token for token in verb.lefts]
        antecedent = None
        for token in lefts:
            if token.dep_ == dep:
                try:
                    # token to the left of present one
                    possible_antecedent = sent[token.i-1]
                    # if there is a child ("relcl" dependency tag)
                    if possible_antecedent.children:
                        antecedent = possible_antecedent
                except:
                    pass

        if antecedent:
            for chunk in sent_chunks:
                if chunk.root.text == antecedent.text:  
                    antecedent = chunk  # get the full antecedent chunk
                    chunk_list.append(antecedent)
                    conjuncts = [conjunct for conjunct in antecedent.conjuncts]
                    for chunk2 in sent_chunks:
                        if chunk2.root in conjuncts:
                            chunk_list.append(chunk2)

        else:
            for chunk in base_list:
                chunk_list.append(chunk)
                conjuncts = [conjunct for conjunct in chunk.root.conjuncts]
                for chunk2 in sent_chunks:
                    if chunk2.root in conjuncts:
                        chunk_list.append(chunk2)

    return chunk_list, preposition

def get_prep_phrases_triples(sent, attribute):
    '''
    add prepositional phrases that do not have the verb as their head (i.e. 'detached' prepositional phrases)
    '''
    sent_chunks = [chunk for chunk in sent.noun_chunks]

    triples = []
    labelled_triples = []

    base_list = [chunk for chunk in sent_chunks if chunk.root.dep_ == "pobj" and chunk.root.head.head in [
        chunk1.root for chunk1 in sent_chunks]]  # exclude prepositions where head is verb or aux (i.e. not noun chunk)

    for chunk in base_list:
        detached_pobjs = []
        detached_pobjs.append(chunk)
        preposition = chunk.root.head
        head = preposition.head
        conjuncts = [conjunct for conjunct in chunk.root.conjuncts]
        for chunk2 in sent_chunks:
            if chunk2.root in conjuncts:
                detached_pobjs.append(chunk2)

        for chunk3 in detached_pobjs:
            triples.append((head, preposition, chunk3))
            labelled_triples.append({'source':head, 
                                     'target':chunk3, 
                                     'edge':  preposition.text.lower(),
                                     'attribute': attribute})
    return triples, labelled_triples

def get_polarity(verb):
    '''
    check if there is a negation or not
    '''
    negations = [child for child in verb.children if child.dep_ == "neg"]
    if negations:
        polarity = 'not '
    else:
        polarity = ''

    return polarity


def get_adjectives(noun_chunk):
    '''
    get adjectives inside a noun chunk (i.e. adjectives that modify the noun chunk head)
    '''
    adjectives = [token.lemma_ for token in noun_chunk if token.pos_ == 'ADJ']
    return adjectives


# main SVO function
def get_SVO_triples(text, nlp, prep_phrases=False, transform_span = lambda x: x, attribute = None):

    """
    Gives subject-predicate-object relations between noun chunks in a sentence

    Takes a text (or doc), returns subject-predicate-object relations between noun chunks in a sentence
    if "prep_phrases" is True, also return prepositional phrases detached from the main verb e.g. ("weapons", "terror", {'edge': "of"})
    attribute is a variable that can be included (e.g. the id of the social media message)
    
    Returns a list of triples (each triple containing spacy token spans) per text/doc


    Parameters
    ----------
    text : string
        text to extract triples from
    nlp : spacy.model
        the Spacy model to use.
    prep_phrases: bool
        whether to incluse prepositional phrases or not
    transform_span: function
        A function to choose what to do with the span (e.g. span.root.lemma_)
        defaults to the identity function
    attribute: Any
        optional attribute to give with each text (such as timestamp)

    Returns
    -------
    
        A list of triples, each triple is a dictionary with source, target, edge and attribute keys.

    Examples
    --------
    >>> get_SVO_triples(text="The book was given to him by the President of the United States.", nlp=nlp, prep_phrases=False)
    [{'source': President, 
      'target': the United States, 
      'edge': 'of', 'attribute': None}, 
     {'source': The book, 
      'target': him, 'edge': 
      'be give by', 
      'attribute': None}, 
     {'source': The book, 
      'target': the President, 
      'edge': 'be give by', 
      'attribute': None}]
    """
    doc = nlp(text) if type(text) != spacy.tokens.doc.Doc else text

    triples = []
    labelled_triples = []

    for sent in doc.sents:
        # add 'detached' prepositional phrases
        if prep_phrases:
            triple, labeled_triple = get_prep_phrases_triples(sent, attribute)
            triples.extend(triple)
            labelled_triples.extend(labeled_triple)

        verbs = [token for token in sent if token.pos_ in ["VERB", "AUX"]]
        #get pos chunks
        for verb in verbs:
            nsubjs, _ = get_pos_chunks(sent, verb, "nsubj")
            dobjs, _ = get_pos_chunks(sent, verb, "dobj")
            nsubjpass, _ = get_pos_chunks(sent, verb, "nsubjpass")
            attrs, _ = get_pos_chunks(sent, verb, "attr")
            datives, _ = get_pos_chunks(sent, verb, "dative")
            pobjs, _ = get_pos_chunks(sent, verb, "pobj")
            agents, _ = get_pos_chunks(sent, verb, "agent")
            _, prep = get_pos_chunks(sent, verb, "pobj")

            polarity = get_polarity(verb)

            def add_triple(nsubj, verb, pos_chunks, format_string):
                for chunk in pos_chunks:
                    triples.append((nsubj, verb, chunk))
                    
                    # transform_span is default a function that does nothing
                    # but can be substituted with a function that transforms a span into a lemma / root / ...
                    labelled_triples.append({'source': transform_span(nsubj),
                                             'target':  transform_span(chunk), 
                                             "edge": format_string(polarity, verb),
                                             "attribute" : attribute})
                    
            # active triples
            for nsubj in nsubjs:
                
                # get nsubj - V - dobj triples
                add_triple(nsubj, verb, dobjs, lambda polarity, verb: polarity + verb.lemma_.lower())
                
                # # get nsubj - V - dative triples
                add_triple(nsubj, verb, datives, lambda polarity, verb:  polarity + verb.lemma_.lower() + ' to')

                # # get nsubj - V - pobj triples
                add_triple(nsubj, verb, pobjs, lambda polarity, verb: polarity + verb.lemma_.lower() + ' ' + prep.text.lower())

                # get nsubj - V - agent triples
                add_triple(nsubj, verb, agents, lambda polarity, verb: polarity + verb.lemma_.lower() + ' ' + prep.text.lower())

                # # get nsubj - V - attr triples
                add_triple(nsubj, verb, attrs, lambda polarity, verb: polarity + verb.lemma_.lower())

            # passive version
            for nsubj in nsubjpass:
                
                # get nsubj - V - dobj triples
                add_triple(nsubj, verb, dobjs, lambda polarity, verb: polarity + 'be ' + verb.lemma_.lower())

                # get nsubj - V - dative triples
                add_triple(nsubj, verb, datives, lambda polarity, verb: polarity + 'be ' + verb.lemma_.lower() + ' to')

                # # get nsubj - V - pobj triples
                add_triple(nsubj, verb, pobjs, lambda polarity, verb: polarity + 'be ' + verb.lemma_.lower() + ' ' + prep.text.lower())

                # # get nsubj - V - agent triples
                add_triple(nsubj, verb, agents, lambda polarity, verb: polarity + 'be ' + verb.lemma_.lower() + ' ' + prep.text.lower())

                # # get nsubj - V - attr triples
                add_triple(nsubj, verb, attrs, lambda polarity, verb: polarity + 'be ' + verb.lemma_.lower())
                
    return labelled_triples