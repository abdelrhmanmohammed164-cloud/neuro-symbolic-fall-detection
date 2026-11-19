# Fold 1 audit + Fold 0 baseline: Two-Stream ST-GCN (scene-disjoint CV)

الهدف من التقرير ده مش إننا نرفع الرقم. الهدف إننا نعرف بالظبط الموديل بيعمل إيه على مشهد عمره ما شافه، وليه بيغلط، وإيه الخطوة الجاية الآمنة علميًا.

قاعدة مهمة طول التقرير: نتايج test بتاعة Fold 1 اتشافت خلاص، فبقت **contaminated** لأي اختيار جاي. اتستخدمت هنا للتشخيص بس. ومفيش threshold ولا hyperparameter اتغيّر بناءً عليها.

كل التحاليل اتعملت على نسخة من الملفات المحفوظة على الماك، والـ checksum بتاعها مطابق. الـ checkpoints مقاسة بـ sha256: `best_checkpoint.pt` = `ff77aa8de0da0476…` و`last_checkpoint.pt` = `dc7e9a33976d3864…`.

---

## 1. التحقق من النتايج الموجودة (Phase 1)

| الفحص | النتيجة |
|---|---|
| حساب الـ metrics من جديد من الـ per-frame probabilities في `result.json` | مطابق بالظبط لـ `result.json` ولـ `fold1_diagnosis.json` وللـ log: TP 502, TN 4695, FP 233, FN 419, F1 0.6063 |
| كل frame في الـ test متقيّم مرة واحدة بس | أيوه، 5849 frame من 29 فيديو، ومفيش ولا صف متكرر |
| الـ labels في `result.json` بتطابق ملفات `_frame_labels.npy` | أيوه، في كل الـ frames |
| الـ best epoch | 44، وهو نفس الـ epoch اللي فيه أقل val loss في `training_history.csv` |
| إعادة الـ diagnosis على CPU من نفس الـ checkpoint | نفس الأرقام بالظبط: train F1 0.952، val F1 0.947، test F1 0.606. يعني النتايج reproducible |
| الـ 15 pre-training test | أعدت تشغيلهم بنفسي على نفس الداتا والـ splits، و15/15 PASS. بس اتنين منهم نطاقهم أضيق من اسمهم (تفاصيل في جزء 4) |

### تصحيح الاستنتاج القديم: CONFIRMED BUG (في الريبورت، مش في الموديل)

`fold1_diagnosis.json` كتب: "test performance is well above baseline and close to training performance". **الكلام ده غلط.** الـ train F1 كان 0.952 والـ test F1 كان 0.606، يعني فرق 0.346. الـ verdict اتكتب كده لأن الـ threshold كان 0.35 والفرق جه أقل منه بشعرة.

اتصلّح في `diagnose_fold.py`. أي فرق أكبر من 0.15 بقى بيطلع حكمه "LEARNS A TRANSFERABLE SIGNAL, BUT WITH A LARGE GENERALISATION GAP"، والأرقام مكتوبة جواه. اتجرب على Fold 1 وطلّع الحكم الصح، وهو محفوظ في `fold1_diagnosis_v2.json`. أما `fold1_diagnosis.json` الأصلي فمتسابش زي ما هو من غير تعديل.

علامة "val loss rising" كانت معتمدة على آخر epoch بس، وده رقم noisy. بقت دلوقتي بتقارن متوسط آخر 5 epochs. الحكم في Fold 1 مااتغيرش (0.094 مقابل أحسن قيمة 0.033).

---

## 2. Root cause analysis (Phase 2)

### A. Overfitting ولا distribution shift؟

- الـ val F1 عند الـ epoch المختار (0.947) قريب جدًا من الـ train F1 (0.952). يعني **جوه نفس المشاهد مفيش overfitting كلاسيكي**. الفجوة بتظهر بس لما ننقل لمشهد جديد. ده بيشاور على **scene/domain shift**، مش على حفظ للـ training videos. **LIKELY EXPLANATION**
- السبب إن الـ validation ما كشفتش المشكلة إنها **من نفس مشاهد التدريب**: `splits.py` بياخد 20% عشوائي من فيديوهات التدريب. فالـ val مش بتقيس أي حاجة عن مشهد جديد، واختيار الـ checkpoint مبني عليها. **CONFIRMED DATA/DESIGN PROBLEM** (كان مكتوب في الـ docstring، بس أثره أكبر مما كنا متوقعين)
- التدريب نفسه مش مستقر. الـ val loss وصل لـ 15.7 في epoch 17، ومن epoch 14 لـ 24 الموديل كان بيقول "fall" على كل حاجة في الـ val (recall 1.0 مع F1 حوالي 0.3). ده غالبًا من Adadelta مع lr 1.0 ومفيش gradient clipping. **LIKELY EXPLANATION** للتذبذب. ومش واضح إنه سبب الفجوة.
- الـ capacity: الموديل فيه 1.03M parameter، وبيتدرّب على 32 فيديو بس (منهم 14 وقعة). capacity كبيرة بالنسبة للداتا. **HYPOTHESIS TO TEST**
- الـ class weighting: الـ `pos_weight` = 2.07، اتحسب من training windows بتاعة epoch 0 بس. ولو حسبناه على كل الـ 50 epoch كان هيتراوح بين 1.95 و2.24. الأخطاء في الـ test أغلبها FN (419) مش FP (233)، فالـ weighting مش بيعمل false positives زيادة. **مش سبب أساسي**

### B. Scene and domain shift

كل مشهد في URFD فيه class واحد بس: F1 وF2 كلها وقعات، وA4 لحد A8 كلها ADL. يعني المشهد والـ label متطابقين. **CONFIRMED DATA PROBLEM** الـ scene-disjoint split بيمنع الموديل يحفظ الأوضة، لكنه مش بيضمن إن مفيش shortcut تاني مشترك بين مشاهد الوقعات وبيفرّقها عن مشاهد الـ ADL، زي ارتفاع الكاميرا أو مكان الجسم في الصورة.

إحصائيات المشاهد (median لكل فيديو، والإحداثيات normalized x/W و y/H):

| Scene | دور في Fold 1 | فيديوهات | طول الفيديو (frames) | missing joints | body scale (neck to hip) | neck y آخر 10 frames |
|---|---|---|---|---|---|---|
| F1 | train/val | 18 fall | 70.5 | 7.6% | 0.132 | 0.92 |
| F2 | **test** | 12 fall | 130 | 4.3% | **0.205** | 0.92 |
| A4 | train/val | 12 ADL | 250 | 10.8% | 0.156 | 0.39 |
| A6 | train/val | 6 ADL | 97.5 | 0.0% | 0.108 | 0.49 |
| A8 | train/val | 5 ADL | 200 | 9.9% | 0.088 | 0.74 |
| A5 | **test** | 11 ADL | 180 | 5.4% | 0.172 | 0.51 |
| A7 | **test** | 6 ADL | 342.5 | 11.6% | 0.117 | **0.89** |

اللي باين من الجدول:

1. **الوقعات في الـ test أطول، وبتبدأ متأخر.** في التدريب كل الوقعات بتبدأ قبل frame 60 (من 4 لـ 59)، والفيديوهات طولها من 55 لـ 110 frame. في الـ test الوقعات بتبدأ من frame 11 لحد frame 148.
2. **الجسم أكبر في الـ test.** 7 من 12 وقعة في الـ test عندهم body scale أكبر من أكبر قيمة شافها الموديل في التدريب (0.207).
3. **المشهد A7 (test) الناس فيه بتخلص الفيديو تحت في الصورة** (neck y = 0.89)، زي الوقعة بالظبط. وده المشهد اللي فيه أكتر frames "lying" مقصودة في URFD (662 frame).

**الـ 4 وقعات الأسوأ** (fall-03 وfall-09 اتفوّتوا خالص، وfall-07 وfall-11 اتمسكوا بعد أكتر من 2.2 ثانية) كلهم body scale فوق 0.24، وكلهم بيبدأوا متأخر أو عندهم lying phase قصيرة. وكل الوقعات اللي scale بتاعها 0.232 أو أقل اتمسكت.

**عملت تجربتين counterfactual على train/val بس (من غير test):**

| التجربة | الفكرة | النتيجة |
|---|---|---|
| `context_probe.py` | ضفت 40 و80 و120 frame قبل الوقعة في فيديوهات train/val، عشان الوقعة تبدأ متأخر | الوقعات فضلت تتمسك (18/18)، والـ recall فضل بين 0.89 و0.92. **الفرضية مش مدعومة** |
| `scale_probe.py` | كبّرت الجسم ×1.25 و×1.5 و×1.8 حوالين الرقبة | الوقعات فضلت تتمسك (18/18). بس عند ×1.8 بدأت false alarms في 11 فيديو ADL. **الفرضية مش مدعومة للوقعات**، والموديل حساس للـ scale في الـ ADL |

يعني ولا التأخير لوحده ولا الحجم لوحده بيكرر الفشل. **التحفظ المهم:** 14 من الـ 18 وقعة في التجربتين دول training videos الموديل حافظها، والتكبير الرقمي مش زي إن واحد يقرّب من الكاميرا فعلًا (perspective، والجسم بيتقص من حواف الصورة). فسبب الـ 4 وقعات دول لسه **NOT VERIFIED**. المرشحين: الـ viewpoint واتجاه الوقعة بالنسبة للكاميرا. والخطوة الجاية لده إننا نبص على الـ RGB frames بتاعتهم بالعين.

### C. Window labelling

| السؤال | الإجابة |
|---|---|
| الـ label بيطابق الهدف؟ | الـ label هو label آخر frame في الـ window (causal)، وده بيطابق per-frame score في Eq. 6. **OK** |
| كام window اتشال من التدريب؟ | على مدار الـ 50 epoch، 6,945 من 37,914 window بتخلص على fall frame (**18.3%**) خدوا weight 0. في epoch 0 لوحده، 143 من 2048 (7%) |
| الاستبعاد بيشيل early-fall بس؟ | أيوه بالتعريف: أي window فيه أقل من 10 fall frames، يعني أول 9 frames بعد بداية الوقعة. **CONFIRMED** (by design) |
| الموديل بيتعلم post-fall posture أساسًا؟ | **LIKELY EXPLANATION**. الدليل: الـ positive windows اللي اتدرب عليها median بتاعها 28 frame بعد البداية (p10 = 12، p90 = 48). والاحتمال بتاعه شكله step: بيقفز من 0 لـ 1 ويفضل 1 طول فترة الـ lying (`fold1_per_video_probs.png`). و191 من 233 FP frame بتقع على frames URFD نفسها معلّمها "lying" مقصود في فيديوهات ADL |
| الـ early detection بيتعاقب؟ | أيوه. الـ evaluation بيحسب أول 9 frames من كل وقعة (108 frame في الـ test)، بينما التدريب عمره ما ادّاهم weight. 79 من الـ 419 FN (19%) جايين من هنا. **CONFIRMED inconsistency بين train وeval** |
| نفس تعريف الـ label في train وeval؟ | نفس التعريف (آخر frame)، والفرق الوحيد هو الـ weight 0 للـ onset في التدريب |
| الـ overlapping windows بتشوّه الـ metrics؟ | الـ eval بيعمل window لكل frame (stride 1)، يعني frames متجاورة شبه متطابقة. الـ 5849 frame **مش** 5849 ملاحظة مستقلة. عشان كده الـ uncertainty محسوبة بالـ bootstrap على مستوى الفيديو (جزء 3) |

مااتغيرش أي حاجة في تعريف الـ labels.

### D. Class imbalance

| | فيديوهات fall / ADL | frames | fall frames |
|---|---|---|---|
| Train | 14 / 18 | 4,876 | 710 (14.6%) |
| Val | 4 / 5 | 1,211 | 172 (14.2%) |
| Test | 12 / 17 | 5,849 | 921 (15.7%) |
| Training windows (epoch 0) | 620 positive / 1,285 negative / 143 weight 0 | | pos_weight 2.07 |

النسب متقاربة في الـ 3 splits. الـ `pos_weight` اتحسب من training windows بس (`train.py` سطر 207 لـ 209)، ومفيش val ولا test labels داخلين فيه. **OK**

---

## 3. تحليل الـ FN والـ FP (Phase 3)

الملفات: `fold1_per_video_errors.csv` و`fold1_per_video_probs.png` و`fold1_errors_per_video.png`.

### على مستوى الوقعة (event level)، وده الأهم

| المقياس | القيمة | 95% CI (bootstrap على الفيديوهات، 5000 مرة) |
|---|---|---|
| وقعات اتمسكت | **10 / 12** (83%) | 58% لحد 100% |
| وقعات اتفوّتت | fall-03، fall-09 | |
| detection latency (median) | 0.58 ثانية | |
| أكبر latency | 2.37 ثانية (fall-11) | |
| ADL videos من غير ولا false alarm | 10 / 17 (59%) | 35% لحد 82% |
| false-alarm episodes في الـ ADL | 12 episode في 2.34 دقيقة من الـ ADL، يعني 5.1 في الدقيقة | |
| frame-level F1 | 0.61 | 0.39 لحد 0.77 |

خلي بالك إن الـ 5.1 إنذار في الدقيقة رقم مضخّم. فيديوهات URFD الـ ADL قصيرة ومتجمّعة حوالين أنشطة صعبة (قعدة، نوم)، فالرقم ده مش هو المعدل الحقيقي في بيت فيه حد عايش يوم عادي.

الـ CI واسعة جدًا. 12 وقعة بس في الـ test ميكفوش عشان أي رقم هنا يبقى دقيق.

### الـ 419 FN مش 419 وقعة

- **مركّزين في 5 فيديوهات:** fall-11 (71) وfall-07 (68) وfall-03 (68) وfall-09 (65) وfall-05 (59)، ودول مع بعض 331 من 419 (79%). أكبر 3 لوحدهم 49%.
- **حسب المرحلة:** 79 في أول 9 frames من الوقعة (اللي التدريب عمره ما ادّالها weight)، و143 في الـ falling phase، و197 في الـ lying phase (URFD labels).
- 3 وقعات بس فيهم 5 FN أو أقل (fall-04 وfall-10 وfall-12)، وfall-08 صفر.

### الـ 233 FP

- **مركّزين في 7 فيديوهات ADL:** adl-30 (75) وadl-37 (55) وadl-38 (54) وadl-10 (26) وadl-40 (11) وadl-36 (4) وadl-39 (3)، وفيه 5 frames قبل بداية الوقعة في fall-08. أكبر 3 = 79%.
- 6 من الـ 7 فيديوهات من المشهد **A7**.
- **191 من 233 FP frame** واقعين على frames URFD معلّمها "lying"، يعني حد نايم بمزاجه. الـ 42 الباقيين في adl-37.
- الـ windows اللي فيها FP فيها **29% missing joints**، مقابل 13% في الـ TN، ومتوسط الـ confidence 0.78 مقابل 0.90.

### Error grouping (متوسط الـ 80-frame window)

| النوع | عدد | missing joints | confidence | padded window | neck y |
|---|---|---|---|---|---|
| TP | 502 | 3.0% | 0.94 | 55% | 0.76 |
| FN | 419 | 4.9% | 0.96 | 18% | 0.69 |
| FP | 233 | **29.2%** | **0.78** | 2% | 0.64 |
| TN | 4695 | 13.3% | 0.90 | 41% | 0.34 |

الـ FN مش بسبب pose رديء، فالـ confidence عالية. الـ TP أغلبها padded windows (55%) مقابل 18% بس في الـ FN. ده بيتماشى مع إن الموديل بيمسك الوقعة أسهل لما تكون بدري في الفيديو، زي ما كانت في التدريب.

---

## 4. Leakage re-audit (Phase 4)

| # | الفحص | الحالة | الدليل |
|---|---|---|---|
| 1 | كل windows الفيديو في نفس الـ split | **OK** | الـ windows بتتبني من قايمة الفيديوهات بتاعة كل split، ومفيش window بيعدّي من فيديو لفيديو |
| 2 | الـ scene groups صح | **OK** | `splits.verify()` واختبار 06. الـ test: A5 وA7 وF2. الـ train/val: A4 وA6 وA8 وF1 |
| 3 | near-duplicates بتعدّي الحدود | **CONFIRMED BUG (val بس)** | في Fold 1، adl-25 وadl-27 في التدريب وadl-29 في الـ val. `NEAR_GROUPS` بيتطبق في video mode بس، مش على الـ val split جوه scene mode. ده بيخلي الـ val متفائلة شوية، بس مش بيأثر على الـ test. اختبار 07 فحص train ضد test بس، عشان كده عدّى |
| 4 | overlapping windows بين الـ splits | **OK** | الـ windows مش بتعدّي حدود الفيديو |
| 5 | الـ normalization من held-out data | **OK** | x/W وy/H تحويل ثابت. الـ BatchNorm running stats جاية من training batches بس |
| 6 | الـ class weights | **OK** | من training windows بس |
| 7 | الـ augmentation | **OK** | `TrainWindows` بس. `EvalWindows` من غير augmentation |
| 8 | val/test labels في الـ preprocessing | **OK** | الـ preprocessing بيشتغل على كل فيديو لوحده ومن غير labels |
| 9 | الـ temperature calibration | **OK** | على الـ val بس. ملحوظة: الـ temperature مش بتغيّر أي قرار عند threshold 0.5 |
| 10 | الـ test في اختيار الـ checkpoint | **OK** | الـ test بيتقيّم مرة واحدة بعد اختيار الـ checkpoint |
| 11 | test-derived statistics في التغييرات الجاية | **RISK، لازم يتوثق** | إحنا الاتنين شفنا نتايج test بتاعة Fold 1، والتحليل كله هنا مبني على فشلها. أي تعديل جاي متحفّز منها يبقى "test-informed"، ولازم يتقيّم على داتا تانية (جزء 7 و9). وفيه نقطة تانية: الـ probes استخدمت فيديوهات F1 كـ training data بتاعة Fold 1، والفيديوهات دي هي **test set بتاع Fold 0**. مفيش أي إعداد في Fold 0 اتغير بسببها |
| 12 | subject independence | **NOT VERIFIABLE** | مفيش subject IDs في URFD عندنا |
| - | الـ padding shortcut | **CONFIRMED DATA PROBLEM (per fold)** | اختبار 15 قاس الـ AUC على كل الـ 70 فيديو (0.501). جوه training set بتاع Fold 1 لوحده: P(padded \| fall) = 0.91 مقابل 0.67 للـ non-fall، والـ AUC بتاع (padded أو لأ) = 0.617. والـ window اللي بيبدأ من frame 0 للـ fall = 0.78 مقابل 0.47. السبب إن كل فيديوهات F1 قصيرة. الـ p_pad بيوازن على مستوى الداتا كلها، مش جوه كل fold |
| - | الـ background shortcut | **SUSPECTED** | الموديل skeleton-only ومش شايف pixels. لكن خصايص المشهد الهندسية (ارتفاع الكاميرا، بعد الشخص، مكانه في الصورة) بتدخل عن طريق x/W وy/H، ومرتبطة بالـ class لأن المشاهد class-pure |

---

## 5. ST-GCN code audit (Phase 5)

| البند | الحالة |
|---|---|
| Graph topology: 14 joint، و13 bone شجرة متصلة حوالين الـ Neck، و3 partitions | **OK** (اتأكدت بالكود + اختبار 02) |
| COCO-17 → 14 mapping | **OK** (L_Shoulder = 5، R_Shoulder = 6 … L_Ankle = 15، R_Ankle = 16، Head = nose، Neck = منتصف الكتفين) |
| Coordinate stream (Eq. 2) | **OK**: (x, y) بس، والـ conf مش داخلة إلا بـ `--use_conf` |
| Motion stream (Eq. 3) | **OK**: backward difference، و0 عند t = 1، و0 لو الـ joint missing في أي frame من الاتنين |
| الترتيب الزمني والـ causality | **OK**: الـ window بيغطي [t−79, t] |
| الـ shapes | **OK**: (N, 3, 80, 14, 1) → logits (N,) |
| Fusion (Eq. 5) | **OK**: concat [h_sp ‖ h_tm] (256) → φ = Linear 128 + ReLU + Dropout 0.5 |
| Classifier (Eq. 6) | **OK**: Linear(128 → 1)، والـ sigmoid جوه BCEWithLogits |
| Gradient flow | **OK**: الاتنين streams بياخدوا gradient |
| Loss | **OK**، بس الـ train loss فيها pos_weight والـ val loss من غيرها. فالمنحنيين مش قابلين للمقارنة في القيمة المطلقة، وده موثّق في الرسم |
| **Missing joints في الـ coordinate stream** | الـ joint الـ missing بيبقى (0, 0)، يعني أعلى شمال الصورة. ده رقم صناعي قوي، والـ FP windows فيها missing joints ضعف الـ TN. **HYPOTHESIS TO TEST** |
| **Padding** | right padding بتكرار آخر frame (زي ما الورقة قالت: "padded with the last valid observation")، والـ motion بتبقى 0 في الجزء المكرر. **OK للورقة**، لكن عامل residual shortcut في Fold 1 (جزء 4) |
| **Normalization** | x/W وy/H (الورقة). مكان الجسم المطلق في الصورة بيعتمد على الكاميرا، فممكن يبقى camera-dependent shortcut. **HYPOTHESIS TO TEST** |
| Motion normalization | مفيش normalization زيادة، يعني مش بتشيل معلومة. بس هي scale-dependent |

### الورقة مقابل اختياراتنا

| حاجة الورقة حددتها | حاجة الورقة ما حددتهاش (اختيارنا) |
|---|---|
| Two streams (coordinate + motion)، Eq. 2 لحد 6 | عدد الـ blocks والـ channels (6 blocks: 32 → 128) |
| ΔP backward difference | φ = Linear 256 → 128، والـ dropout |
| Graph بـ spatial + temporal edges | الـ 14-joint mapping من COCO |
| Window 80، والـ padding بآخر observation | label = آخر frame، وweight 0 للـ onset |
| Adadelta، 50 epoch، batch 256 | weight decay 1e-4، والـ pos_weight، والـ temperature، واختيار الـ checkpoint بالـ val loss |
| x/W، y/H | augmentation (flip وscale 0.9 لـ 1.1 وshift وjitter)، والـ p_pad 0.35 |

---

## 6. Fold 0 ومقارنة الاتجاهين (Phase 6)

Fold 0 اتشغّل بنفس الإعدادات بالظبط (نفس الـ config fingerprint `6f9d66e1a7b3`، ونفس الـ split والداتا). التعديل الوحيد قبله كان تصليح نص الـ verdict في `diagnose_fold.py`، ومش بيلمس التدريب. الـ 15 اختبار عدّوا تاني قبل التدريب. والـ sha256 بتاع checkpoints Fold 1 طلع هو هو قبل التشغيل وبعده (`ff77aa8de0da0476` و`dc7e9a33976d3864`).

**إفصاح:** test set بتاع Fold 0 (F1 + A4 + A6 + A8) ما اتقيّمش عليه أي موديل Fold 0 قبل كده، وما اتغيرش أي إعداد بسببه. بس فيديوهات F1 كانت training data بتاعة Fold 1، والـ probes في جزء 2 استخدمتها، وشفنا نتيجة Fold 1 قبل ما نشغّل Fold 0. الملفات: `fold0_audit.json` و`fold0_per_video_errors.csv` و`fold0_per_video_probs.png` و`fold0_missing_probe.json`.

### الأرقام

| | Fold 0 (test: F1 + A4 + A6 + A8) | Fold 1 (test: F2 + A5 + A7) | Pooled out-of-fold |
|---|---|---|---|
| Train F1 / Val F1 (نفس المشاهد) | 0.967 / 0.932 | 0.952 / 0.947 | |
| **Test frame F1** (95% CI) | **0.53** (0.45 لحد 0.61) | **0.61** (0.39 لحد 0.77) | **0.555** |
| Test recall / specificity | 0.91 / **0.74** | **0.55** / 0.95 | 0.725 / 0.842 |
| TP / FN / FP / TN | 805 / 77 / **1366** / 3839 | 502 / **419** / 233 / 4695 | 1307 / 496 / 1599 / 8534 |
| Accuracy مقابل "always non-fall" | **0.763 مقابل 0.855** | 0.889 مقابل 0.843 | **0.824 مقابل 0.849** |
| وقعات اتمسكت | **18 / 18** | 10 / 12 | **28 / 30** (CI 83% لحد 100%) |
| Median latency | 0.02 ثانية | 0.58 ثانية | 0.08 ثانية |
| ADL videos من غير ولا false alarm | **1 / 23** | 10 / 17 | **11 / 40** (CI 15% لحد 43%) |
| Best epoch | 32 | 44 | |

### المقارنة

1. **المشكلة موجودة في الاتجاهين، بس بشكلين عكس بعض.** Fold 1 بيفوّت أو بيتأخر (FN). Fold 0 بيمسك كل الوقعات بس بيصرّخ على أي حاجة (FP). يعني الموضوع مش مشهد واحد "صعب". الموديل بيتعلم حاجة خاصة بمشاهد التدريب، وبتتقلب على أي مشهد جديد.
2. **الـ accuracy مضللة.** في Fold 0 والـ pooled، الـ accuracy أقل من موديل بيقول "مفيش وقعة" على طول. وده بيأكد إن الـ accuracy مينفعش تبقى دليل على أي حاجة هنا.
3. **في الاتجاهين، الـ in-scene validation كانت بتقول 0.93 أو أكتر.** فالـ validation الحالية مش بتتنبأ بالأداء على مشهد جديد خالص.

### الـ false alarms في Fold 0

- **منتشرة، مش مركّزة:** في 22 من 23 فيديو ADL، وأكبر 3 فيديوهات = 28% بس (في Fold 1 كانوا 79%). 38 episode في 2.6 دقيقة ADL.
- **84% منهم على ناس واقفة أو بتتحرك عادي** (URFD label "not lying")، و11% بس على lying مقصود. ده عكس Fold 1 (80% على lying).
- **أول 79 frame في كل فيديو** (الـ windows اللي فيها padding): نسبة الـ FP في الـ ADL 0.40 لحد 0.48، مقابل 0.18 بعد frame 80. في Fold 1 نفس الـ windows دي كانت 0.00. يعني الـ padding بيأثر على الموديل بشكل بيتقلب حسب الـ fold. **CONFIRMED** (القياس)، والسبب **NOT VERIFIED**
- **مشهد A4:** كل فيديوهاته تقريبًا (adl-12 لحد adl-21) بيطلع فيها alarm بيبدأ في أول 3 لـ 17 frame ويقف عند frame 82 لحد 103 تقريبًا، يعني بعد ما أول ~25 frame يخرجوا من الـ window. الفيديوهات دي بتبدأ و6 لـ 8 joints من 14 missing.
- **اختبرت فرضية الـ missing joints (diagnostic بس):** ملّيت الـ joints الناقصة بأقرب قيمة معروفة وشغلت نفس الـ checkpoint. الـ alarms ما اختفتش (من 0.44 لـ 0.47 في أول 80 frame). **الفرضية اترفضت.** اللي بيعمل الـ alarm هو حاجة الـ joints الظاهرة بتعملها في أول الفيديو، ومحتاجين نشوف الـ RGB عشان نعرفها.
- **ملاحظة structural:** حدث واحد قصير في أي مكان جوه الـ 80 frame بيخلي الاحتمال عالي لحد ما يخرج من الـ window. ده متماشي مع الـ global average pooling على الزمن مع label آخر frame. **LIKELY EXPLANATION** لطول الـ false-alarm episodes (حوالي 80 frame).

### الـ FN في Fold 0

77 بس: 55 منهم في أول 9 frames من الوقعة (اللي التدريب بيدّيها weight 0)، و22 في الـ falling phase. مفيش ولا FN في الـ lying phase.

---

## 7. خطة تحسين صغيرة ومتحكم فيها (Phase 7)

**تحذير صريح:** أغلب الأفكار دي متحفّزة من فشل test بتاع Fold 1. يعني لو جربناها واخترنا أحسنها بناءً على Fold 1 تاني، يبقى بنعمل tuning على الـ test. عشان كده كل تجربة لازم تتقيّم بـ inner validation منفصل بالمشهد، وتتأكد في الآخر على داتا خارجية.

الترتيب حسب الأولوية، وتعديل واحد في كل مرة:

| # | التعديل | الدافع والدليل | المخاطرة | إزاي يتقيّم من غير test |
|---|---|---|---|---|
| 0 | **إصلاح الـ evaluation design الأول** (مش تعديل في الموديل) | الـ val الحالي من نفس المشاهد، وفيه near-duplicate (adl-29)، فمش بيقيس أي حاجة عن scene shift | لو ما اتعملش، أي اختيار جاي هيبقى على أساس رقم مضلل | inner leave-one-ADL-scene-out جوه التدريب. والـ fall scene الوحيد في التدريب يتقسم بالفيديو |
| 1 | Gradient clipping (مثلًا max norm 1.0)، مع نفس Adadelta | الـ val loss وصل لـ 15.7 وكان بيقول fall على كله 10 epochs | ممكن يبطّأ التعلم | ثبات الـ inner-val loss عبر الـ epochs، وتكرار بـ 3 seeds |
| 2 | Padding: نقيس الأول early frames لوحدهم، وبعدين نجرب left padding (الـ frames الحقيقية في آخر الـ window، والتكرار في أوله) بدل right padding | أول 79 frame بيتصرفوا بشكل مختلف تمامًا في الـ fold-ين، وده من غير ما نعرف السبب | الورقة بتقول "padded with the last valid observation"، فالتغيير ده engineering choice ولازم يتوثق | ADL false alarms في أول 80 frame على الـ inner val |
| 3 | Body-centred + body-scale normalization (`window_center` + القسمة على طول الـ torso) | الـ test فيه جسم أكبر وأماكن تانية في الصورة، والمشاهد class-pure | ممكن نشيل معلومة الارتفاع المطلق اللي مفيدة فعلًا لما الشخص ينزل. والـ symbolic stream هيعوّض ده في الـ fusion | inner val + نفس الـ probes على train/val |
| 4 | Missing-joint handling (آخر قيمة معروفة، أو `--use_conf`). أولويته قلّت لأن الـ probe في Fold 0 رفض إنه سبب alarms مشهد A4 |  الـ FP في Fold 1 فيها missing joints ضعف الـ TN | الـ conf نفسها ممكن تبقى scene-dependent | ADL false alarms على الـ inner val |
| 4b | Per-fold padding balancing (p_pad يتظبط لحد ما الـ padded fraction تتساوى بين الـ classes جوه كل fold) | AUC 0.617 جوه Fold 1 | قليلة | اختبار 15 يتعدّل ويتعمل per fold |
| 5 | Hard negatives: الـ ADL lying frames تاخد weight أعلى | 191 من 233 FP على lying مقصود | ممكن تقل الـ recall | ADL false alarms + recall على inner val |

**مش هنعمل دلوقتي:** تغيير الـ threshold، أو تغيير الـ architecture، أو تغيير تعريف الـ labels. الحاجات دي مكانها بعد ما الـ evaluation تتصلح، أو في الـ fusion مع الـ symbolic stream. ده هو بالظبط اللي الورقة عاملاه عشان تفرّق الوقعة عن النوم المقصود.

---

## 8. Checkpoints وأمان التجارب (Phase 8)

- ملفات Fold 1 (الـ checkpoints، والـ history، و`result.json`، و`fold1_diagnosis.json`) مااتلمستش. الـ script الجديد بيطبع الـ sha256 بتاع الـ checkpoints قبل التشغيل وبعده.
- `run_fold.sh` بيرفض يكتب على diagnosis موجود، وبيكتب في `fold<k>/` بس.
- الـ resume بتاع التدريب (atomic writes، وfingerprints، وRNG state، وbest/last checkpoint) اتجرب في اختبار 12 و13، والاتنين PASS في إعادة التشغيل بتاعتي.
- كل تجربة تحسين لازم تتعمل في `out_dir` جديد (مثلًا `runs/exp01_gradclip/`). والـ `train.py` بيرفض يكمل أو يكتب فوق نتايج إعداداتها مختلفة.

---

## 9. تصميم الـ evaluation النهائي (Phase 9)

1. **Outer:** leave-scene-out بالـ fall scene (2 folds)، زي ما هو دلوقتي.
2. **Inner (لاختيار الـ checkpoint وأي hyperparameter):** جوه التدريب، ADL scene كامل يتساب برّه + جزء من فيديوهات الـ fall scene، مع احترام `NEAR_GROUPS`. ده nested CV على قد ما الداتا تسمح.
3. **الوحدة الإحصائية:** الفيديو أو الوقعة، مش الـ frame. نبلّغ event recall، والـ latency، وعدد الـ ADL videos اللي فيها false alarm، ومعاهم bootstrap CI على الفيديوهات.
4. **التفرقة في الكتابة:** training، ثم inner validation، ثم cross-validation held-out (Fold 0 + Fold 1)، ثم external test.
5. **Cross-validation مش بديل لـ external test.** لازم داتا مستقلة، زي Le2i أو Multiple Cameras Fall بعد ما يتعملهم نفس الـ preprocessing، عشان أي claim عن الـ generalization.
6. **Subject-disjoint:** مش ممكن من غير subject IDs، ومش هنقول إن عندنا ده.
7. الأرقام بتاعتنا على URFD المعدّلة **مش reproduction مباشر** لنتايج الورقة، ومش هتتقارن بيها كأنها نفس الحاجة.

---

## 10. ترتيب المشاكل حسب الخطورة

| # | المشكلة | التصنيف |
|---|---|---|
| 1 | الـ validation من نفس مشاهد التدريب، فالـ checkpoint بيتختار على رقم مش بيقيس scene shift. في الـ fold-ين كانت ≥ 0.93 والـ test 0.53 و0.61 | CONFIRMED DATA PROBLEM (design) |
| 1b | الفشل في الاتجاهين بشكلين عكس بعض: Fold 1 بيفوّت (FN)، وFold 0 false alarms في 22 من 23 ADL. والـ accuracy المجمعة (0.824) أقل من "always non-fall" (0.849) | CONFIRMED (القياس) |
| 1c | أول 79 frame في الفيديو (padded windows) بيتصرفوا بشكل مختلف تمامًا حسب الـ fold: ADL FP 0.40 لحد 0.48 في Fold 0، و0.00 في Fold 1 | CONFIRMED (القياس)، والسبب NOT VERIFIED |
| 2 | URFD scenes class-pure: المشهد = الـ label | CONFIRMED DATA PROBLEM |
| 3 | الموديل بيتصرف كـ "lying posture detector": احتماله step، و82% من الـ FP على lying مقصود | LIKELY EXPLANATION |
| 4 | الفجوة بين train وtest (0.95 مقابل 0.61) جاية من scene shift مش overfitting جوه المشهد | LIKELY EXPLANATION |
| 5 | الـ 4 وقعات الأسوأ: scale كبير + بداية متأخرة، بس الـ counterfactuals البسيطة ما كررتش الفشل | NOT VERIFIED (السبب لسه مش معروف) |
| 6 | residual padding shortcut جوه Fold 1 (AUC 0.617)، واختبار 15 بيقيس على الداتا كلها بس | CONFIRMED DATA PROBLEM + test coverage gap |
| 7 | near-duplicate adl-29 في الـ val ومعاه adl-25/27 في التدريب | CONFIRMED BUG (val only) |
| 8 | التدريب مش مستقر (val loss 15.7، و10 epochs بيقول fall على كله) | LIKELY EXPLANATION (Adadelta lr 1.0 من غير clipping) |
| 9 | الـ onset frames: التدريب بيدّيلها weight 0 والـ eval بيحسبها (79 FN) | CONFIRMED inconsistency (by design) |
| 10 | Missing joints = (0, 0) في الـ coordinate stream | **اترفضت** كسبب لـ alarms مشهد A4 في Fold 0. لسه HYPOTHESIS TO TEST للـ FP في Fold 1 |
| 11 | x/W وy/H مطلقين، فممكن camera shortcut | HYPOTHESIS TO TEST |
| 12 | نص الـ verdict في `diagnose_fold.py` كان غلط | CONFIRMED BUG، **اتصلح واتجرب** |
| 13 | subject independence | NOT VERIFIABLE |
| 14 | مفيش external test set | CONFIRMED DATA PROBLEM |

---

## 11. أوامر التشغيل

```bash
# Fold 0 بالإعدادات المجمّدة
bash ~/Downloads/University\ project/scripts/run_fold.sh 0

# إعادة التحليل (من فولدر الكود، جوه الـ venv)
cd ~/Downloads/University\ project/fall_detection_models/code/stgcn_urfd
python analyze_fold.py ~/Downloads/University\ project ~/Downloads/University\ project/reports/scene_cv_audit . 1   # أو 0
python context_probe.py ~/Downloads/University\ project . ~/Downloads/University\ project/reports/scene_cv_audit/fold1_context_probe.json
python scale_probe.py   ~/Downloads/University\ project . ~/Downloads/University\ project/reports/scene_cv_audit/fold1_scale_probe.json
```

---

## 12. القرار

**REDESIGN EVALUATION**

الموديل مفيهوش bug بيخلّيه يغلط. الـ architecture مطابق للورقة، ومفيش leakage للـ test، والأرقام reproducible. اللي فيه مشكلة هو طريقة اختيارنا للـ checkpoint وطريقة قياسنا للـ generalization. الـ validation الحالية من نفس المشاهد، ومش ممكن نختار بيها أي تحسين من غير ما نرجع نبص على الـ test.

**الترتيب:**
1. ~~نخلّص Fold 0 بالإعدادات المجمّدة~~ اتعمل. الفشل موجود في الاتجاهين (جزء 6)، وده بيأكد القرار.
2. نصلّح الـ validation: inner validation منفصلة بالمشهد، وتحترم `NEAR_GROUPS`. ونخلي اختبار 15 per fold، واختبار 07 يغطي train ضد val. وده تعديل في الـ evaluation بس، مش في الموديل.
3. بعدها بس نبدأ تجارب التحسين بالترتيب اللي في جزء 7، تعديل واحد في كل مرة، وكل تجربة في فولدر لوحدها.
4. بالتوازي: نبص على RGB frames بتاعة fall-03 وfall-09 وfall-07 وfall-11 (Fold 1)، وعلى أول 30 frame في فيديوهات A4 (Fold 0)، عشان نعرف سبب الفشل.
5. نبلّغ الـ metrics لأول 79 frame ولباقي الفيديو كل واحد لوحده، عشان الـ padding ما يخبيش حاجة.
6. external dataset قبل أي claim عن الـ generalization.

---

## 13. اللي اتصلح بعد الـ audit

### صور الـ RGB للفيديوهات اللي فشلت (`rgb/`)

دي كانت أهم خطوة، لأن التجارب الرقمية الـ 3 (التأخير، والحجم، والـ missing joints) ما فسّرتش الفشل.

| الفيديو | اللي باين في الصور | التصنيف |
|---|---|---|
| fall-09، fall-07 (اتفوّتت أو اتأخرت) | الشخص بيقع **ناحية الكاميرا**، فجسمه وهو نايم بيبقى على امتداد محور الكاميرا. في الصورة 2D الجسم بيبان قصير ومش أفقي، وراسه قريبة من حافة الصورة تحت | LIKELY EXPLANATION |
| fall-03 (اتفوّتت) | الشخص قريب جدًا من الكاميرا وراسه برّه الصورة قبل الوقعة، وبعدين بيقع على جنبه وهو لسه قريب | LIKELY EXPLANATION |
| fall-11 (اتأخرت 2.4 ثانية) | وقعة جانبية، بس فيها مرحلة طويلة بيتقلب فيها ورجليه فوق قبل ما ينام مفرود. الموديل مسكها لما اتفرد بس | LIKELY EXPLANATION |
| fall-12 (اتمسكت فورًا) | وقعة جانبية، والجسم موازي لمستوى الصورة | (للمقارنة) |
| fall-13 (تدريب Fold 1) | الشخص بعيد عن الكاميرا وصغير. فيه شكل تاني (شخص أو مانيكان) قاعد على الشمال في كل الـ frames، بس الـ skeleton عمره ما نط عليه (مفيش ولا jump في الـ neck x في أي فيديو) | اتفحص، ومفيش مشكلة |
| adl-12 (A4، alarms Fold 0) | **الأوضة ضلمة تقريبًا**، والشخص يادوب باين في الباب. الـ confidence في أول 25 frame من 0.61 لـ 0.78، مقابل 0.88 لـ 0.97 بعد كده | CONFIRMED (الإضاءة وجودة الـ pose) |
| adl-30 (A7، FP في Fold 1) | واحد راكع على الأرض قدام الكاميرا ونص جسمه متقص تحت | CONFIRMED hard negative |

الخلاصة إن الفشل مش bug. **الداتا اللي اتدرب عليها الموديل ما فيهاش زوايا وقعات وإضاءة زي اللي اتختبر عليها.** URFD فيها كاميرا واحدة لكل مشهد، ووقعات كل مشهد بنفس الاتجاه تقريبًا. وده بيقوّي إن external data (كاميرات وزوايا مختلفة) شرط عشان أي claim عن التعميم.

### التصليحات (كلها اتجربت)

| # | المشكلة (جزء 10) | التصليح | الدليل |
|---|---|---|---|
| 1 | الـ validation من نفس المشاهد | `splits.py --val_mode scene` → `split_scene_v2.json`: مشهد ADL كامل برّه التدريب (القاعدة اتحددت قبل التجارب: أصغر مشهد. Fold 0 → A7، وFold 1 → A8)، ومعاه جزء عشوائي من الوقعات | الـ test folds نفسها بالظبط، والـ split القديم مااتغيرش (fingerprint `17fe4100648f`) |
| 7 | near-duplicate في الـ val | `keep_near_groups_together`، و`split_video_v2.json` بـ `--val_mode random_grouped` | اختبار 07 الجديد **بيفشل على الـ splits القديمة** (scene وvideo الاتنين)، وبيعدّي على v2 |
| 6 | اختبار 15 كان بيقيس على الداتا كلها بس | 15 = الداتا كلها (gate)، و16 جديد = لكل fold (WARN مش بيوقف التدريب) | اختبار 16 على v2: Fold 0 = 0.62 مقابل 0.67 (AUC 0.48)، وFold 1 = 0.94 مقابل 0.70 (**AUC 0.62، WARN**) |
| 6 | الـ padding shortcut جوه الـ fold | `--pad_balance` (experiment، مش default) | Fold 1: من 0.94 مقابل 0.70 لـ 0.94 مقابل 0.95، والـ AUC من 0.62 لـ 0.49 |
| 11 | أي نتيجة test بتأثر على الاختيار | `--no_test` بيكتب `val_result.json` بس، واختبار 17 بيثبت إن الـ test عمره ما اتقيّم | 17 PASS، والـ test videos متحطوط فيها NaN |
| 8 | تدريب مش مستقر | `--grad_clip` (experiment) | |
| 1c | الـ padding بيتصرف بشكل مختلف | `--pad_side left` (experiment) | |

**الـ default مااتغيرش:** الـ default path في `dataset.py` بيطلّع نفس الـ windows ونفس الـ tensors بالظبط زي الكود القديم (اتأكدت بالمقارنة المباشرة). والـ config fingerprint بتاع الإعدادات الافتراضية لسه `6f9d66e1a7b3`، فنتايج الـ baseline القديمة فاضلة صالحة ومتربطة بالكود.

### التجارب (`scripts/run_experiments.sh`)

كلها على `split_scene_v2.json` ومع `--no_test`. تعديل واحد في كل تجربة:

| التجربة | التعديل |
|---|---|
| baseline_v2_s42، baseline_v2_s43 | الإعدادات الأصلية بـ seed-ين، عشان نقيس الـ noise |
| exp01_gradclip | `--grad_clip 1.0` |
| exp02_leftpad | `--pad_side left` |
| exp03_useconf | `--use_conf` (الـ confidence كـ channel تالت) |
| exp04_center | `--norm_mode window_center` |
| exp05_padbalance | `--pad_balance` |

**قاعدة القرار، اتكتبت قبل أي تجربة** (في أول `compare_experiments.py`): التجربة تتقبل بس لو في **الـ fold-ين الاتنين**:
- الـ val F1 ≥ الـ baseline + max(0.02، فرق الـ seed-ين)،
- ومفيش وقعة اتفوّتت زيادة،
- ومفيش فيديو ADL زيادة فيه false alarm.

غير كده تترفض. التجارب اللي تتقبل تتجمع في run جديد لازم يعدّي نفس القاعدة. **تحفظ:** الـ val صغيرة جدًا (Fold 0: وقعتين + 6 ADL، وFold 1: 4 وقعات + 5 ADL)، فالقاعدة مش هتلقط غير التأثيرات الكبيرة.

**بعد التجارب:** الإعداد اللي هيتختار يتشغّل **مرة واحدة** على الـ test بتاع الـ fold-ين، في نفس الفولدر من غير إعادة تدريب (الـ resume اتجرب). ويتكتب صراحة إن test بتاع Fold 1 وFold 0 كان اتشاف قبل كده، فالأرقام دي "post-hoc" ومش تقييم مستقل. التقييم المستقل الوحيد هيبقى على external dataset.

### حاجات لسه ما اتعملتش، عن قصد

- **Missing-joint imputation:** الـ probe رفضها كسبب لـ alarms مشهد A4، فاتأجلت.
- **Hard-negative weighting للـ ADL lying:** ده هيتعمل في الـ fusion مع الـ symbolic stream (Eq. 8 لـ 10 في الورقة بتفرّق النزول السريع عن النوم المقصود)، مش في الـ neural stream لوحده.
- **رفع الـ threshold أو تغييره:** ممنوع، لأنه هيتظبط على test اتشاف.
- **External dataset:** محتاج تحميل وpreprocessing جديد. ده الخطوة الجاية بعد التجارب.

---

## 14. القرار الحالي

القرار في جزء 12 كان **REDESIGN EVALUATION**، وده اتعمل. القرار دلوقتي:

**READY FOR CONTROLLED IMPROVEMENT EXPERIMENTS**، بشرط:
- التجارب تتعمل على الـ inner validation الجديدة بس، وبالقاعدة اللي اتكتبت قبلها.
- مفيش test number يتشاف لحد ما الإعداد النهائي يتختار.
- بعدها **COLLECT ADDITIONAL DATA**، يعني external dataset بكاميرات وزوايا مختلفة. صور الـ RGB وضّحت إن الفشل جاي من زوايا وإضاءة مش موجودة في التدريب، وده مش هيتحل بأي tuning على URFD لوحده.

---

## 15. التجربة الصغيرة (10 epochs) وتغيير قاعدة اختيار الـ checkpoint

قبل ما نشغّل التجارب الكاملة، عملنا تجربة صغيرة (`run_experiments.sh quick`، 10 epochs، الـ baseline و`pad_balance` بس). الملفات في `runs/exp_v2_quick/` و`reports/experiments_v2_quick/`، وفاضلين زي ما هما.

**اللي طلع:** الـ pipeline شغّال على الماك (16 PASS، و1 WARN متوقع، وصفر FAIL، ومفيش ولا test number). **بس قاعدة اختيار الـ checkpoint بالـ val loss بقت بايظة مع الـ validation الجديدة.** أقل val loss على مشهد ما اتشافش بييجي في أول 0 لـ 2 epochs، والموديل ساعتها بيقول "مفيش وقعة" على كله (F1 = 0). ولما الموديل بيبدأ يتعلم (Fold 1 الـ val F1 وصل 0.43 في epoch 9)، الـ val loss بيطلع، لأنه بيدّي false alarms واثق منها على المشهد الجديد. يعني لو كنا شغّلنا التجارب الكاملة زي ما هي، كل الموديلات المختارة كانت هتبقى موديلات مش بتعمل حاجة.

**ليه مش F1:** في test صغير، epoch قال "fall" على كل حاجة (recall 1.0، specificity 0.0) وأخد F1 = 0.315. فالـ F1 بيكافئ الموديل اللي بيقول fall على طول.

**التصليح:** `--select val_mcc`، يعني الـ checkpoint اللي عنده أعلى **MCC** على الـ validation، ولو اتساووا ناخد الأقل في val loss. الـ MCC بيبقى صفر لأي موديل بيقول نفس الحاجة على كله، fall أو non-fall. ومطابق لـ sklearn. واختبار 18 بيتأكد إنه بيختار الـ epoch الصح، وإن الـ resume بيدّي نفس النتيجة بالظبط.

القرار ده اتاخد من **validation curves بس**، من غير أي test data، وقبل أي تجربة حقيقية. وقاعدة المقارنة في `compare_experiments.py` اتحدثت لنفس الـ metric (MCC بدل F1)، وده قبل التجارب برضه. التجارب الجديدة بتتحفظ في `runs/exp_mcc/`، عشان ما تتخلطش بالتجربة الصغيرة الأولى.
