# Divorce Predictors Scale - Question Mappings
# Based on the divorce dataset attributes (cleaned version uses Atr2, Atr3, ..., Atr54)
# Original scale: 0 (Never) to 4 (Always)

DIVORCE_QUESTIONS = {
    'Atr2': "When we need it, we can take our discussions with my spouse from the beginning and correct it.",
    'Atr3': "I know we can ignore our differences, even if things get hard sometimes.",
    'Atr4': "When we need it, we can discuss our disagreements with my spouse.",
    'Atr6': "My spouse and I have similar ideas about how marriage should be.",
    'Atr7': "My spouse and I have similar values in trust.",
    'Atr10': "I know my spouse's basic anxieties.",
    'Atr13': "When I'm arguing with my spouse, contacting him/her will eventually work.",
    'Atr22': "I enjoy traveling with my spouse.",
    'Atr23': "Most of our goals for people (children, friends, etc.) are the same.",
    'Atr24': "I think that one day in the future, when I look back, I see that my spouse and I have been in harmony with each other.",
    'Atr28': "My spouse and I have similar ideas about how roles should be in marriage.",
    'Atr31': "My spouse and I have similar ideas about being free.",
    'Atr32': "We don't have time at home as partners.",
    'Atr34': "We're like two strangers who share the same environment at home rather than family.",
    'Atr42': "I enjoy my holidays with my spouse.",
    'Atr43': "I enjoy traveling with my spouse.",
    'Atr44': "Most of our goals are common to my spouse.",
    'Atr45': "I think one day in the future, when I look back, I see that my spouse and I have been in harmony with each other.",
    'Atr46': "My spouse and I have similar values in terms of personal freedom.",
    'Atr47': "My spouse and I have similar entertainment.",
    'Atr48': "Most of our goals for people (children, friends, etc.) are the same.",
    'Atr49': "Our dreams with my spouse are similar and harmonious.",
    'Atr50': "We're compatible with my spouse about what love should be.",
    'Atr51': "We share the same views with my spouse about being happy in our life.",
    'Atr52': "My spouse and I have similar ideas about how marriage should be.",
    'Atr53': "My spouse and I have similar ideas about how roles should be in marriage.",
    'Atr54': "My spouse and I have similar values in trust."
}

# Column names for divorce_clean.csv
CLEAN_COLUMNS = ['Atr2', 'Atr3', 'Atr4', 'Atr6', 'Atr7', 'Atr10', 'Atr13', 'Atr22', 'Atr23', 'Atr24', 
                 'Atr28', 'Atr31', 'Atr32', 'Atr34', 'Atr42', 'Atr43', 'Atr44', 'Atr45', 'Atr46', 
                 'Atr47', 'Atr48', 'Atr49', 'Atr50', 'Atr51', 'Atr52', 'Atr53', 'Atr54']

SCALE_LABELS = {
    0: "Never",
    1: "Seldom",
    2: "Sometimes",
    3: "Often",
    4: "Always"
}

def format_couple_features(feature_values):
    """Format couple features as question-answer pairs."""
    lines = []
    for col, val in zip(CLEAN_COLUMNS, feature_values):
        question = DIVORCE_QUESTIONS.get(col, f"Question {col}")
        scale_label = SCALE_LABELS.get(int(val), str(val))
        lines.append(f"Q: {question}\nA: {val} ({scale_label})")
    return "\n\n".join(lines)
