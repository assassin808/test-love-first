# Speed Dating Dataset - Complete Field Documentation

**Dataset**: Speed Dating Data.csv  
**Total Records**: 8,378  
**Total Columns**: 195  
**Unique Participants**: 551  

---

## 📋 Field Categories & Descriptions

### 🆔 **Basic Identifiers (1-13)**

| # | Field | Coverage | Description |
|---|-------|----------|-------------|
| 1 | `iid` | 100.0% | **Unique subject ID** - grouped by (wave, id, gender) |
| 2 | `id` | 100.0% | **Subject number within wave** |
| 3 | `gender` | 100.0% | **Gender** - 0=Female, 1=Male |
| 4 | `idg` | 100.0% | **Subject number within gender** - grouped by (id, gender) |
| 5 | `condtn` | 100.0% | **Condition** - 1=limited choice, 2=extensive choice |
| 6 | `wave` | 100.0% | **Wave number** - 21 waves total (2002-2004) |
| 7 | `round` | 100.0% | **Number of people met in wave** |
| 8 | `position` | 100.0% | **Station number where met partner** |
| 9 | `positin1` | 78.0% | **Station number where started** |
| 10 | `order` | 100.0% | **The number of date that night when met partner** |
| 11 | `partner` | 100.0% | **Partner's id number the night of event** |
| 12 | `pid` | 99.9% | **Partner's iid number** |
| 13 | `match` | 100.0% | **Match result** - 1=yes (both said yes), 0=no |

---

### 👥 **Partner Information (14-33)**

| # | Field | Coverage | Description |
|---|-------|----------|-------------|
| 14 | `int_corr` | 98.1% | **Interest correlation** - correlation between participant's and partner's interests |
| 15 | `samerace` | 100.0% | **Same race indicator** - 1=yes, 0=no |
| 16 | `age_o` | 98.8% | **Partner's age** |
| 17 | `race_o` | 99.1% | **Partner's race** |
| 18-23 | `pf_o_att/sin/int/fun/amb/sha` | 98.5-98.9% | **Partner's stated preferences** (Time 1) - attr/sinc/intel/fun/amb/shar |
| 24 | `dec_o` | 100.0% | **Partner's decision** - 1=yes, 0=no |
| 25-30 | `attr_o/sinc_o/intel_o/fun_o/amb_o/shar_o` | 87.2-97.5% | **Partner's ratings of you** (Scorecard) |
| 31 | `like_o` | 97.0% | **How much partner liked you** (1-10) |
| 32 | `prob_o` | 96.2% | **Partner's estimate of your yes** (1-10) |
| 33 | `met_o` | 95.4% | **Partner met you before?** - 1=yes, 2=no |

---

### 👤 **Demographics & Background (34-50)**

| # | Field | Coverage | Description |
|---|-------|----------|-------------|
| 34 | `age` | 98.9% | **Your age** |
| 35 | `field` | 99.2% | **Field of study** (text) |
| 36 | `field_cd` | 99.0% | **Field coded** - 1=Law, 2=Math, 3=Social Science, 4=Medical, 5=Engineering, 6=English/Writing, 7=History/Religion/Philosophy, 8=Business/Econ, 9=Education, 10=Bio/Chem/Physics, 11=Social Work, 12=Undergrad/undecided, 13=Political Science, 14=Film, 15=Fine Arts, 16=Languages, 17=Architecture, 18=Other |
| 37 | `undergra` | 58.7% | **Undergraduate school** |
| 38 | `mn_sat` | 37.4% | **Median SAT score** of undergraduate institution (proxy for intelligence) |
| 39 | `tuition` | 42.8% | **Tuition** of undergraduate institution |
| 40 | `race` | 99.2% | **Your race** - 1=Black/African American, 2=European/Caucasian-American, 3=Latino/Hispanic, 4=Asian/Pacific Islander, 5=Native American, 6=Other |
| 41 | `imprace` | 99.1% | **Importance of same race** in dating (1-10) |
| 42 | `imprelig` | 99.1% | **Importance of same religion** in dating (1-10) |
| 43 | `from` | 99.1% | **Where from originally** (before Columbia) |
| 44 | `zipcode` | 87.3% | **Zip code where grew up** |
| 45 | `income` | 51.1% | **Median household income** based on zipcode |
| 46 | `goal` | 99.1% | **Primary goal** - 1=Fun night out, 2=Meet new people, 3=Get a date, 4=Serious relationship, 5=Say I did it, 6=Other |
| 47 | `date` | 98.8% | **Dating frequency** - 1=Several times/week, 2=Twice/week, 3=Once/week, 4=Twice/month, 5=Once/month, 6=Several times/year, 7=Almost never |
| 48 | `go_out` | 99.1% | **Go out frequency** (not necessarily dates) - Same scale as `date` |
| 49 | `career` | 98.9% | **Intended career** (text) |
| 50 | `career_c` | 98.4% | **Career coded** - 1=Lawyer, 2=Academic/Research, 3=Psychologist, 4=Doctor/Medicine, 5=Engineer, 6=Creative Arts/Entertainment, 7=Banking/Consulting/Finance/Business, 8=Real Estate, 9=International/Humanitarian Affairs, 10=Undecided, 11=Social Work, 12=Speech Pathology, 13=Politics, 14=Pro sports/Athletics, 15=Other, 16=Journalism, 17=Architecture |

---

### 🎯 **Interests & Activities (51-67)** - Rate 1-10

| # | Field | Coverage | Description |
|---|-------|----------|-------------|
| 51 | `sports` | 99.1% | **Playing sports/athletics** |
| 52 | `tvsports` | 99.1% | **Watching sports** |
| 53 | `exercise` | 99.1% | **Body building/exercising** |
| 54 | `dining` | 99.1% | **Dining out** |
| 55 | `museums` | 99.1% | **Museums/galleries** |
| 56 | `art` | 99.1% | **Art** |
| 57 | `hiking` | 99.1% | **Hiking/camping** |
| 58 | `gaming` | 99.1% | **Gaming** |
| 59 | `clubbing` | 99.1% | **Dancing/clubbing** |
| 60 | `reading` | 99.1% | **Reading** |
| 61 | `tv` | 99.1% | **Watching TV** |
| 62 | `theater` | 99.1% | **Theater** |
| 63 | `movies` | 99.1% | **Movies** |
| 64 | `concerts` | 99.1% | **Going to concerts** |
| 65 | `music` | 99.1% | **Music** |
| 66 | `shopping` | 99.1% | **Shopping** |
| 67 | `yoga` | 99.1% | **Yoga/meditation** |

---

### 🎭 **TIME 1: Pre-Event Survey (68-97)**

#### Expectations (68-69)
| # | Field | Coverage | Description |
|---|-------|----------|-------------|
| 68 | `exphappy` | 98.8% | **Expected happiness** - how happy expect to be (1-10) |
| 69 | `expnum` | 21.5% | ⚠️ **Expected matches** - out of 20 people, how many expect to be interested |

#### 1️⃣ **Self Preferences** (70-75) - 100 points to distribute
| # | Field | Coverage | Description |
|---|-------|----------|-------------|
| 70 | `attr1_1` | 99.1% | **Attractive** - importance in potential date |
| 71 | `sinc1_1` | 99.1% | **Sincere** - importance in potential date |
| 72 | `intel1_1` | 99.1% | **Intelligent** - importance in potential date |
| 73 | `fun1_1` | 98.9% | **Fun** - importance in potential date |
| 74 | `amb1_1` | 98.8% | **Ambitious** - importance in potential date |
| 75 | `shar1_1` | 98.6% | **Shared interests** - importance in potential date |

#### 4️⃣ **What Same Sex Looks For** (76-81) - 100 points
| # | Field | Coverage | Description |
|---|-------|----------|-------------|
| 76 | `attr4_1` | 77.5% | ⚠️ **Attractive** - what fellow men/women look for |
| 77 | `sinc4_1` | 77.5% | ⚠️ **Sincere** |
| 78 | `intel4_1` | 77.5% | ⚠️ **Intelligent** |
| 79 | `fun4_1` | 77.5% | ⚠️ **Fun** |
| 80 | `amb4_1` | 77.5% | ⚠️ **Ambitious** |
| 81 | `shar4_1` | 77.2% | ⚠️ **Shared interests** |

#### 2️⃣ **What Opposite Sex Looks For** (82-87) - 100 points
| # | Field | Coverage | Description |
|---|-------|----------|-------------|
| 82 | `attr2_1` | 99.1% | **Attractive** - what opposite sex looks for |
| 83 | `sinc2_1` | 99.1% | **Sincere** |
| 84 | `intel2_1` | 99.1% | **Intelligent** |
| 85 | `fun2_1` | 99.1% | **Fun** |
| 86 | `amb2_1` | 98.9% | **Ambitious** |
| 87 | `shar2_1` | 98.9% | **Shared interests** |

#### 3️⃣ **Self-Ratings** (88-92) - Rate 1-10
| # | Field | Coverage | Description |
|---|-------|----------|-------------|
| 88 | `attr3_1` | 98.7% | **Attractive** - how I rate myself |
| 89 | `sinc3_1` | 98.7% | **Sincere** |
| 90 | `fun3_1` | 98.7% | **Fun** |
| 91 | `intel3_1` | 98.7% | **Intelligent** |
| 92 | `amb3_1` | 98.7% | **Ambitious** |

#### 5️⃣ **Others' Perception** (93-97) - Rate 1-10
| # | Field | Coverage | Description |
|---|-------|----------|-------------|
| 93 | `attr5_1` | 58.6% | ⚠️ **Attractive** - how others would rate me |
| 94 | `sinc5_1` | 58.6% | ⚠️ **Sincere** |
| 95 | `intel5_1` | 58.6% | ⚠️ **Intelligent** |
| 96 | `fun5_1` | 58.6% | ⚠️ **Fun** |
| 97 | `amb5_1` | 58.6% | ⚠️ **Ambitious** |

---

### ⭐ **SCORECARD: During Event (98-108)**
*Filled out after each 4-minute date*

| # | Field | Coverage | Description |
|---|-------|----------|-------------|
| 98 | `dec` | 100.0% | ✅ **Your decision** - 1=yes, 0=no |
| 99 | `attr` | 97.6% | ✅ **Attractive** - rating of partner (1-10) |
| 100 | `sinc` | 96.7% | ✅ **Sincere** |
| 101 | `intel` | 96.5% | ✅ **Intelligent** |
| 102 | `fun` | 95.8% | ✅ **Fun** |
| 103 | `amb` | 91.5% | ⚠️ **Ambitious** (bottleneck) |
| 104 | `shar` | 87.3% | ⚠️ **Shared interests** |
| 105 | `like` | 97.1% | ✅ **Overall like** (1-10) |
| 106 | `prob` | 96.3% | ✅ **Probability partner says yes** (1-10) |
| 107 | `met` | 95.5% | ✅ **Met before?** - 1=yes, 2=no |
| 108 | `match_es` | 86.0% | **Match estimate** - how many matches expect |

---

### 🔄 **HALF-WAY: Mid-Event Survey (109-119)**
*Filled out halfway through the event*

| # | Field | Coverage | Description |
|---|-------|----------|-------------|
| 109-114 | `attr1_s` to `shar1_s` | 48.9% | ⚠️ **Updated preferences** (1-10 scale) |
| 115-119 | `attr3_s` to `amb3_s` | 47.7% | ⚠️ **Updated self-ratings** (1-10) |

---

### 📅 **TIME 2: Day After Event (120-156)**
*Filled out the day after participating*

#### Overall Satisfaction (120-122)
| # | Field | Coverage | Description |
|---|-------|----------|-------------|
| 120 | `satis_2` | 89.1% | **Satisfaction** - how satisfied with people met (1-10) |
| 121 | `length` | 89.1% | **4 minutes was:** - 1=Too little, 2=Too much, 3=Just right |
| 122 | `numdat_2` | 88.7% | **Number of dates was:** - 1=Too few, 2=Too many, 3=Just right |

#### 7️⃣ **Actual Decision Weights** (123-128) - 100 points
| # | Field | Coverage | Description |
|---|-------|----------|-------------|
| 123-128 | `attr7_2` to `shar7_2` | 23.3-23.7% | ⚠️ **Retrospective weights** - what actually drove decisions |

#### Updated Preferences & Perceptions (129-156)
| # | Field | Coverage | Description |
|---|-------|----------|-------------|
| 129-134 | `attr1_2` to `shar1_2` | 88.9-89.1% | **Self preferences** (updated) |
| 135-140 | `attr4_2` to `shar4_2` | 68.9% | **Same sex preferences** (updated) |
| 141-146 | `attr2_2` to `shar2_2` | 68.9% | **Opposite sex preferences** (updated) |
| 147-151 | `attr3_2` to `amb3_2` | 89.1% | **Self-ratings** (updated) |
| 152-156 | `attr5_2` to `amb5_2` | 52.2% | **Others' perception** (updated) |

---

### 📆 **TIME 3: 3-4 Weeks After (157-195)**
*Filled out after receiving matches*

#### Follow-up Actions (157-161)
| # | Field | Coverage | Description |
|---|-------|----------|-------------|
| 157 | `you_call` | 47.4% | **How many matches you contacted** |
| 158 | `them_cal` | 47.4% | **How many matches contacted you** |
| 159 | `date_3` | 47.4% | **Been on date with matches?** - 0=No, 1=Yes |
| 160 | `numdat_3` | 17.9% | **How many matches dated** |
| 161 | `num_in_3` | 8.0% | **Number in relationship** |

#### Long-term Reflections (162-195)
| # | Field | Coverage | Description |
|---|-------|----------|-------------|
| 162-167 | `attr1_3` to `shar1_3` | 47.4% | **Self preferences** (final reflection) |
| 168-173 | `attr7_3` to `shar7_3` | 24.1% | **Actual decision weights** (final) |
| 174-179 | `attr4_3` to `shar4_3` | 35.3% | **Same sex preferences** (final) |
| 180-185 | `attr2_3` to `shar2_3` | 24.1-35.3% | **Opposite sex preferences** (final) |
| 186-190 | `attr3_3` to `amb3_3` | 47.4% | **Self-ratings** (final) |
| 191-195 | `attr5_3` to `amb5_3` | 24.1% | **Others' perception** (final) |

---

## 🎯 **Key Insights for Filtering**

### ✅ **High Coverage Fields (>95%)** - RELIABLE
- Demographics: `age`, `gender`, `field_cd`, `career_c`, `race` (98-100%)
- Interests: All 17 activities (99.1%)
- Preferences (Time 1): `attr1_1` to `shar1_1`, `attr2_1` to `shar2_1` (98-99%)
- Self-ratings (Time 1): `attr3_1` to `amb3_1` (98.7%)
- Scorecard: `dec`, `attr`, `sinc`, `intel`, `fun`, `like` (95-100%)
- Background: `imprace`, `imprelig`, `goal`, `date`, `go_out`, `exphappy` (98-99%)

### ⚠️ **Medium Coverage Fields (50-90%)** - USE WITH CAUTION
- Same sex preferences (Time 1): `attr4_1` series (77%)
- Others' perception (Time 1): `attr5_1` series (58.6%)
- Scorecard: `amb` (91.5%), `shar` (87.3%)
- Time 2 data: Most fields (88-89%)
- Half-way data: (47-49%)

### ❌ **Low Coverage Fields (<50%)** - RISKY
- `expnum` (21.5%) - Expected number of matches
- Time 2 retrospective weights: `attr7_2` series (23%)
- Time 3 data: Most fields (24-47%)
- Education: `mn_sat` (37%), `tuition` (43%)

---

## 💡 **Recommendation for Experiment**

### **Core Features to Use (98-99% coverage)**:
1. **Demographics**: `age`, `gender`, `field_cd`, `career_c`, `race`
2. **Background Attitudes**: `imprace`, `imprelig`, `goal`, `date`, `go_out`
3. **Expectations**: `exphappy` ✅
4. **Self Preferences**: `attr1_1`, `sinc1_1`, `intel1_1`, `fun1_1`, `amb1_1`, `shar1_1`
5. **Opposite Sex Preferences**: `attr2_1`, `sinc2_1`, `intel2_1`, `fun2_1`, `amb2_1`, `shar2_1`
6. **Self-Ratings**: `attr3_1`, `sinc3_1`, `intel3_1`, `fun3_1`, `amb3_1`
7. **Interests**: All 17 activities
8. **Scorecard**: `dec`, `attr`, `sinc`, `intel`, `fun`, `amb`, `shar`, `like`
9. **Ground Truth**: `match`

### **Optional Features (if needed)**:
- Same sex preferences: `attr4_1` series (77% coverage)
- Others' perception: `attr5_1` series (58.6% coverage)

### **Avoid**:
- Time 2/3 retrospective data (回忆数据)
- `expnum` (too low coverage)
- Education proxies (SAT, tuition - too many missing values)

---

## 📝 **Summary Statistics**

- **Total fields**: 195
- **High coverage (>95%)**: 89 fields
- **Medium coverage (50-95%)**: 45 fields  
- **Low coverage (<50%)**: 61 fields
- **Ground truth available**: 100%
- **Recommended for use**: 50-60 fields

**Generated**: 2025-11-03
