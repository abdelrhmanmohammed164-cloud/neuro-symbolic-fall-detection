# تقرير مراجعة الـ Neural Stream (ST-GCN) قبل التدريب

**القرار: CONDITIONALLY READY**

التدريب ممكن يبدأ، بس بشروط. لقيت مشكلتين خطيرتين في الداتا والكود، واتحلّوا. فاضل حاجتين ماينفعش يتحلّوا من الداتا اللي عندنا، ولازم يتكتبوا مع أي نتيجة:

1. URFD مافيهاش subject IDs، فمقدرش أثبت إن نفس الشخص مش موجود في الـ train والـ test.
2. مفيش test set مستقل. التقييم كله cross-validation.

الأدلة كلها في الملفات اللي في نفس الفولدر: `dataset_validation_report.csv` و `dataset_audit_summary.json` و `scene_groups.csv` و `pretraining_tests.json`.

---

## Phase 1: متطلبات الورقة (Section 3.3.1 و Section 4)

| المتطلب | في الورقة | النوع | التطبيق عندنا |
|---|---|---|---|
| عدد الـ joints | J = 14: 13 joint + neck (نص المسافة بين الكتفين) | Explicit | 14 |
| Coordinate stream | X = (x, y) لكل joint ولكل frame (Eq. 2) | Explicit | (x, y). الـ confidence مش داخلة إلا لو شغّلت `--use_conf` |
| Motion stream | ΔP_t = P_t − P_{t−1}، وصفر عند t = 1 (Eq. 3) | Explicit | نفس التعريف بالظبط، وصفر لو الـ joint ناقص |
| الـ Graph | edges مكانية حسب الجسم، و edges زمنية بين نفس الـ joint في frames متتالية (Eq. 4) | Explicit، بس قايمة الـ edges مش مكتوبة | 13 bone، والـ neck هو المركز |
| الدمج | h = φ(h_sp ‖ h_tm)، كل stream ليه ST-GCN لوحده (Eq. 5) | Explicit | فرعين ST-GCN، وبعدين concat، وبعدين φ |
| الـ Classifier | p_N = σ(wᵀh + b) (Eq. 6) | Explicit | Linear + sigmoid |
| طول الـ sequence | T = 80، sliding window، والـ clip القصيرة بتتعمل لها padding بآخر frame | Explicit | نفس الكلام |
| الـ Normalization | x/W و y/H | Explicit | نفس الكلام، transform ثابت |
| الـ Optimizer، والـ epochs، والـ batch | Adadelta و 50 و 256 | Explicit | دي القيم الـ default |
| الـ Labels | per-frame، binary | Explicit | per-frame |
| عدد الـ layers والـ channels، والـ loss، والـ lr، والـ weight decay، والـ τ_N و L_N | مش مكتوبين | Undocumented | engineering choices، كلها مكتوبة في الكود |
| الـ Split في الورقة | 80/20 على داتا synthetic، والـ test على فيديوهات حقيقية | Explicit | مش ممكن عندنا. استخدمنا CV (Phase 4) |

**اكتشاف مهم:** الكود اللي كان موجود كان stream واحد، والورقة بتقول streamين. اتصلح (`model.py`، `TwoStreamSTGCN`).

---

## A. جاهزية الداتا

| الفحص | النتيجة |
|---|---|
| العدد | 70 من 70 (30 fall و 40 ADL)، مفيش حاجة ناقصة ولا زيادة |
| القراءة، والـ shapes، والـ dtype | كلهم بيتقروا. (3, T, 14, 1) و float32 |
| القيم | كلها finite، والـ coords في [0, 1]، والـ confidence ≤ 1 |
| الـ Missing joints | متمثّلة بصفر في كل الـ channels، ومتطابقة مع conf = 0 |
| Sequences كلها أصفار | 0 |
| Frames متكررة بالظبط | 0 |
| أقل حركة | adl-25 و adl-27 و adl-29 (قعدة هادية). طبيعي، ومش تكرار |
| ترتيب الـ joints | أقصى فرق عن RTMO الخام 3.12 px، والـ neck 1.15 px. الـ mapping مظبوط |
| ترتيب الجسم في الـ frames اللي الشخص فيها واقف | median 93.9%. أقلهم adl-34 بـ 28%، وده لأن الشخص قاعد أو منحني (طوله في الـ Kinect حوالي 1 m)، مش غلط |
| الـ Labels | الـ fall frames مطابقة لـ URFD (label 0 و 1) في الـ 30 fall، ومفيش ولا fall frame في أي ADL. المجموع 1,803 من 11,936 frame |
| التتبع للمصدر | كل sequence ليها `_meta.json` فيه الـ SHA-256 بتاع الـ zip الأصلي |
| الـ Timestamps | adl-37 بس محسوبة من رقم الـ frame. كمان ملفين `adl-02-data.csv` و `adl-03-data.csv` متطابقين على السيرفر (الـ skeletons مختلفة، فده غلط في الـ csv بس) |

---

## B. الـ Leakage

| الخطر | طريقة التحقق | النتيجة | الخطورة | التصحيح |
|---|---|---|---|---|
| **Scene شغّال كـ shortcut** | depth map أول frame، 1-NN | **FAIL**: الخلفية لوحدها بتفصل fall عن ADL 100%، والـ 8 مجموعات scenes كلها من class واحد. الـ skeleton نفسه بيكشف الأوضة: مكان الرجلين في أول الفيديو AUC 0.85، والارتفاع في الصورة 0.76، والـ neck x 0.71 | CRITICAL | **اتحل في التقييم:** الـ split الأساسي بالـ scene، يعني أوض الـ test مابتظهرش في الـ train. الـ shortcut مش هيفيد على الـ test |
| **الـ Padding شغّال كـ shortcut** | نسبة الـ windows اللي فيها padding حسب الـ label | **FAIL في الكود القديم**: 37% من الـ fall windows مقابل 2% بس من الـ non-fall (11 من الـ 12 clip القصيرة falls) | CRITICAL | **اتحل:** padding augmentation للنوعين. بعد التصحيح AUC بتاع عدد الـ padded frames بقى 0.501 (test 15) |
| Source-video leakage | الـ split بالـ sequence | PASS: كل الـ windows بتاعة الفيديو في نفس الجزء (test 06) | | |
| Subject leakage | مفيش subject IDs في URFD | **NOT VERIFIABLE**: فيه 5 ممثلين مجهولين | HIGH | لازم يتكتب. لو اتعرف مين في أنهي فيديو، ممكن نعمل split بالـ subject |
| Temporal overlap | الـ windows المتداخلة من فيديو واحد | PASS: كلها جوه نفس الجزء | | |
| Duplicates | hash، و MAE بعد resampling | PASS: 70 hash مختلفين. أقرب زوج adl-25 و adl-27 (0.0125)، ومش متكررين، بس اتحطّوا في نفس الجزء احتياطاً (test 07) | | |
| Preprocessing leakage | فحص الكود + test بيحط NaN في الـ test fold | PASS: x/W و y/H transform ثابت. الـ pos_weight من الـ train بس، والـ temperature من الـ val بس، والـ BatchNorm من الـ train بس. الـ training والـ checkpoint selection والـ temperature فضلوا finite مع إن الـ test كله NaN (test 08) | | |
| Label leakage من الطول | AUC للطول | WARNING: طول الفيديو بيفصل (AUC 0.09)، بس الموديل مابيشوفش الطول، بيشوف windows بطول 80 بس | MEDIUM | تقييم per-frame، والـ sequence-level تقرير ثانوي |
| Label leakage من الـ missing pose | AUC | PASS: نسبة الـ joints الموجودة مابتفرّقش (AUC 0.53) | | |
| Test contamination | فحص الكود | PASS: الـ test بيتقيّم مرة واحدة في الآخر، ومش داخل في اختيار أي حاجة. **ملحوظة:** قبل المراجعة شغّلت smoke test لـ 3 epochs على الداتا الحقيقية، علشان أتأكد إن الكود شغال بس، ومااتخدش منه ولا قرار | LOW | |
| Augmentation بيعدّي حدود الـ split | الكود | PASS: الـ augmentation في `TrainWindows` بس | | |

---

## C. التوافق مع الورقة

الـ architecture دلوقتي بقت مطابقة لـ Eq. 2 لـ 6. الفروق اللي مافيش منها مفر:

1. **الداتا:** الورقة دربت على synthetic وعملت test على surveillance حقيقي فيه 5 أشخاص. إحنا URFD، فيها شخص واحد و 70 فيديو.
2. **الـ Split:** الورقة 80/20 synthetic مع test حقيقي منفصل. إحنا عاملين leave-scene-out CV بـ 2 folds، لأن URFD فيها أوضتين بس للـ falls، فكل fold بيتدرّب على 10 لـ 14 fall بس من أوضة واحدة.
3. **الـ Pose:** RTMO-l body7، والـ variant مش مكتوب في الورقة.
4. **التفاصيل اللي مش مكتوبة في الورقة واخترناها:** 6 blocks لكل stream بـ channels من 32 لـ 64 لـ 128، و φ = Linear(256→128)، و BCE مع pos_weight، و weight decay 1e-4، وأحسن epoch بالـ val loss، و temperature scaling.
5. **الـ Labels:** الـ fall frame عندنا هو URFD label 0 أو 1 جوه فيديو fall، يعني بيشمل وقت الوقوع ووقت ما الشخص نايم بعده. الورقة مابتقولش هل "نايم بعد الوقعة" بيتحسب fall ولا لأ.

**النتايج ماينفعش تتقارن بـ Table 4 كأنها reproduction.**

---

## D. الـ Training pipeline

**الـ bugs اللي اتصلحت:**
* الموديل كان stream واحد. بقى streamين زي Eq. 5.
* الـ padding shortcut.
* الـ window label ماكانش متسق مع التقييم per-frame. بقى label آخر frame، والـ windows اللي الوقعة لسه بادئة فيها (أقل من 10 fall frames) مستبعدة من الـ loss.
* الـ splits كانت بتتعمل جوه `train.py` في كل مرة. بقت ملف محفوظ بـ fingerprint، وبيرفض يتكتب عليه.
* الـ metrics كانت ناقصة: ضفت specificity و FPR و FNR.

**الـ Checkpoints:**
* `last_checkpoint.pt` بيتحفظ كل epoch، و `best_checkpoint.pt` بيتحفظ بس لما الـ val يتحسن.
* كمان `training_history.csv` و `training.log`.
* كل حاجة بتتحفظ atomic، ومعاها fingerprints للـ config والـ data والـ split.
* الـ resume بيرجّع آخر epoch خلص بالظبط. لو أي fingerprint اختلف، بيرفض يكمّل ومابيبدأش من الأول من غير ما يقول. مابيحفظش وسط الـ epoch.

**الاختبارات:** 15 من 15 PASS (`pretraining_tests.json`). أهمها:
* الـ resume بعد ما وقّفته بيطلّع weights وكل الـ probabilities متطابقة بالظبط مع تشغيل من غير مقاطعة.
* الـ tiny subset اتعلمه 100%، يعني الموديل قادر يتعلّم.
* الـ metrics مطابقة لـ sklearn.

**قيود:**
* على الـ GPU بتاع الماك (MPS)، الـ resume ممكن يختلف في آخر رقم عشري، لأن بعض عمليات الـ GPU مش deterministic تماماً. اختبار التطابق التام اتعمل على CPU.
* الـ temperature بتتحسب من حوالي 6 لـ 9 val sequences بس.

---

## E. القرار

**CONDITIONALLY READY.** الشروط:

1. التقرير الأساسي هو الـ **scene-disjoint split** (`runs/stgcn_scene`). الـ video split (`runs/stgcn_video`) متفائل، لأن الأوض مشتركة بين الـ train والـ test، فيتعرض كـ upper bound بس.
2. يتكتب بوضوح إن الـ subject leakage NOT VERIFIABLE، وإن مفيش test set مستقل.
3. ماتغيّرش أي hyperparameter بناءً على نتايج الـ test. لو عايز تعمل tuning، اعمله على الـ val بس.
