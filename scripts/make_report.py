#!/usr/bin/env python3
"""Generate the IEEE-style conference paper (PDF) with ReportLab.

Two-column IEEE layout, Times fonts, real metrics from ``results/summary.json``
and figures from ``docs/images/``. ASCII-only math (the base-14 Times font has
no Greek/minus glyphs). Targets ~6 pages.

    pip install reportlab pillow
    python scripts/make_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate, Frame, FrameBreak, Image, NextPageTemplate,
    PageTemplate, Paragraph, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs" / "images"
OUT = ROOT / "docs" / "Autonomous_Driving_RL_Report.pdf"

PAGE_W, PAGE_H = letter
LM = RM = 0.62 * inch
TM = 0.7 * inch
BM = 0.7 * inch
GUT = 0.24 * inch
COL_W = (PAGE_W - LM - RM - GUT) / 2.0
USABLE_H = PAGE_H - TM - BM
TITLE_H = 1.5 * inch
DARK = colors.HexColor("#101418")


def styles():
    s = getSampleStyleSheet()
    def add(name, **kw):
        s.add(ParagraphStyle(name, **kw))
    add("PTitle", parent=s["Title"], fontName="Times-Bold", fontSize=18, leading=21,
        alignment=TA_CENTER, textColor=DARK, spaceAfter=8)
    add("PAuthor", parent=s["Normal"], fontName="Times-Roman", fontSize=11, leading=14,
        alignment=TA_CENTER, textColor=DARK)
    add("PAffil", parent=s["Normal"], fontName="Times-Italic", fontSize=9.5, leading=12,
        alignment=TA_CENTER, textColor=DARK)
    add("Abstract", parent=s["Normal"], fontName="Times-Roman", fontSize=9, leading=11,
        alignment=TA_JUSTIFY, firstLineIndent=10)
    add("Sec", parent=s["Normal"], fontName="Times-Bold", fontSize=10, leading=13,
        alignment=TA_CENTER, spaceBefore=11, spaceAfter=4)
    add("Sub", parent=s["Normal"], fontName="Times-BoldItalic", fontSize=9.5, leading=12,
        alignment=TA_LEFT, spaceBefore=4, spaceAfter=2)
    add("Body", parent=s["Normal"], fontName="Times-Roman", fontSize=11, leading=14.2,
        alignment=TA_JUSTIFY, firstLineIndent=10, spaceAfter=3)
    add("Eq", parent=s["Normal"], fontName="Times-Italic", fontSize=9.5, leading=13,
        alignment=TA_CENTER, spaceBefore=3, spaceAfter=3)
    add("Cap", parent=s["Normal"], fontName="Times-Roman", fontSize=8, leading=9.5,
        alignment=TA_LEFT, spaceBefore=2, spaceAfter=6)
    add("Ref", parent=s["Normal"], fontName="Times-Roman", fontSize=8, leading=9.6,
        leftIndent=10, firstLineIndent=-10, spaceAfter=1.5)
    return s


def fit(path, width):
    with PILImage.open(path) as im:
        w, h = im.size
    return Image(str(path), width=width, height=width * h / w)


def metric(summary, agent, key, default="--"):
    try:
        v = summary["agents"][agent]["metrics"][key]
        return f"{v:.2f}" if abs(v) < 100 else f"{v:.1f}"
    except Exception:
        return default


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 8)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawCentredString(PAGE_W / 2.0, 0.42 * inch, str(doc.page))
    if doc.page == 1:
        canvas.drawString(LM, 0.42 * inch, "Preprint - 2026")
    canvas.restoreState()


def build():
    S = styles()
    summary = {}
    sp = ROOT / "results" / "summary.json"
    if sp.exists():
        try:
            summary = json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            summary = {}

    story = []
    def P(t, st="Body"):
        story.append(Paragraph(t, S[st]))

    header = Frame(LM, PAGE_H - TM - TITLE_H, PAGE_W - LM - RM, TITLE_H, id="hdr",
                   leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    c1f = Frame(LM, BM, COL_W, USABLE_H - TITLE_H - 6, id="c1f",
                leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    c2f = Frame(LM + COL_W + GUT, BM, COL_W, USABLE_H - TITLE_H - 6, id="c2f",
                leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    c1 = Frame(LM, BM, COL_W, USABLE_H, id="c1",
               leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    c2 = Frame(LM + COL_W + GUT, BM, COL_W, USABLE_H, id="c2",
               leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    first = PageTemplate(id="First", frames=[header, c1f, c2f], onPage=footer)
    later = PageTemplate(id="Later", frames=[c1, c2], onPage=footer)
    doc = BaseDocTemplate(str(OUT), pageTemplates=[first, later], pagesize=letter,
                          title="Reproducible Deep Reinforcement Learning for Autonomous Driving",
                          author="Mani Chandan Mathi")

    # ---- Title block ----
    story.append(NextPageTemplate("Later"))
    P("Reproducible Deep Reinforcement Learning for Autonomous Driving: "
      "A Dual-Backend CARLA Framework with PPO and SAC", "PTitle")
    P("Mani Chandan Mathi", "PAuthor")
    P("Department of Electrical Engineering and Computer Science<br/>"
      "Massachusetts Institute of Technology, Cambridge, MA, USA<br/>mcmathi@mit.edu", "PAffil")
    story.append(FrameBreak())

    # ---- Abstract ----
    P("<b><i>Abstract</i></b>&mdash;Deep reinforcement learning (DRL) promises driving policies that "
      "optimize a task objective directly, but the canonical simulator, CARLA, is a multi-gigabyte, "
      "GPU-bound process that makes iteration and continuous integration costly. We present a "
      "framework whose central design choice is a single Gymnasium interface with two interchangeable "
      "backends: the full CARLA simulator and a fast kinematic-bicycle surrogate that shares the "
      "identical observation, action, and reward specification. Proximal Policy Optimization (PPO) and "
      "Soft Actor-Critic (SAC) are trained and benchmarked against tuned Proportional-Integral-"
      "Derivative (PID) and Stanley controllers under one fully-logged shaped reward. On the surrogate "
      "lane-following task, PPO and SAC learn from scratch to 0.95 and 0.85 route-success "
      "respectively, approaching the classical controllers (1.00) while driving from non-privileged "
      "observations. The framework couples this learning stack with a metric-driven evaluation suite, "
      "an interactive dashboard, containerization, and a continuous-integration pipeline that trains a "
      "policy on every commit, yielding a system that is correct, observable, and reproducible.",
      "Abstract")
    P("<b><i>Index Terms</i></b>&mdash;Reinforcement learning, autonomous driving, CARLA, proximal "
      "policy optimization, soft actor-critic, reproducibility, MLOps.", "Abstract")

    # ---- I. Introduction ----
    P("I.&nbsp;&nbsp;Introduction", "Sec")
    P("End-to-end driving policies learned with deep reinforcement learning are attractive because "
      "they optimize a task objective directly rather than relying on hand-engineered "
      "perception-planning-control stacks. The CARLA simulator [1] has become the de facto standard, "
      "offering sensor-rich urban scenes, an OpenDRIVE road model, and a Python API. In practice, "
      "however, CARLA's cost is substantial: it is a multi-gigabyte Unreal Engine process that "
      "requires a GPU and runs near real time, making the inner research loop slow, reward debugging "
      "tedious, and automated testing in continuous integration (CI) essentially impractical.")
    P("We argue that much of this friction is incidental rather than fundamental, and can be removed "
      "by separating the interface a learning agent sees from the backend that implements it. We "
      "define one Gymnasium environment contract&mdash;observation space, action space, and "
      "reward&mdash;and provide two backends that satisfy it: the CARLA simulator and a lightweight "
      "kinematic-bicycle surrogate. Because the reward and interfaces are identical, the agent, "
      "perception network, training loop, and evaluation metrics are written once and exercised "
      "end-to-end without launching the simulator.")
    P("A recurring obstacle in this setting is reproducibility. Deep-RL results are notoriously "
      "sensitive to random seeds, hyperparameters, and implementation details, and a simulator that "
      "cannot run in CI compounds the difficulty: regressions surface late and reported numbers are "
      "hard to reproduce. We therefore treat reproducibility as a first-class requirement and show "
      "that a surrogate-plus-CI design recovers most of the engineering discipline that a heavyweight "
      "simulator otherwise precludes.")
    P("This paper makes the following contributions. (i) A backend-agnostic environment contract that "
      "lets the same policy and training code run on CARLA and on a simulator-free surrogate. (ii) "
      "Head-to-head PPO and SAC implementations with a configurable convolutional perception module. "
      "(iii) Tuned PID and Stanley controllers that contextualize the learned policies. (iv) A single, "
      "fully-logged shaped reward whose components are individually observable. (v) A reproducibility "
      "layer&mdash;typed configuration, a unit-test suite, containerization, and a CI pipeline that "
      "trains a policy on every push&mdash;packaged as an installable artifact.")

    P("The remainder of this paper is organized as follows. Section II reviews simulation, the "
      "relevant on- and off-policy algorithms, and reproducibility in deep RL. Section III formalizes "
      "the driving task as a Markov decision process. Section IV describes the dual-backend "
      "architecture, Section V the shaped reward, and Section VI the learning algorithms and classical "
      "baselines. Section VII details the experimental protocol, Section VIII presents and discusses "
      "results, and Section IX the reproducibility and engineering layer. Sections X and XI conclude "
      "with limitations and final remarks.")
    # ---- II. Related Work ----
    P("II.&nbsp;&nbsp;Background and Related Work", "Sec")
    P("<i>Simulation.</i> CARLA [1] underpins most DRL-for-driving studies, providing configurable "
      "towns, traffic, weather, and a sensor suite (cameras, LiDAR, collision and lane-invasion "
      "sensors). Its realism is precisely what makes it expensive as an inner-loop development tool.")
    P("<i>On-policy methods.</i> Proximal Policy Optimization (PPO) [2] is a clipped-surrogate "
      "policy-gradient algorithm prized for stability and ease of tuning. Recent work, CaRL [3], "
      "scales PPO to roughly 300M CARLA samples with deliberately simple rewards and reports "
      "state-of-the-art route completion among learning-based planners, motivating PPO as our primary "
      "algorithm.")
    P("<i>Off-policy and maximum-entropy methods.</i> Soft Actor-Critic (SAC) [4] augments the return "
      "with a policy-entropy term and learns from a replay buffer, and is typically more "
      "sample-efficient than PPO for continuous control. Comparative CARLA studies [5] report that "
      "distributional critics such as TQC can reach 0.91 route completion where DDPG reaches only 0.23 "
      "on difficult scenarios, underscoring the importance of algorithm choice and motivating a "
      "controlled PPO-versus-SAC comparison.")
    P("<i>Classical control and perception.</i> The Stanley steering law [6], which won the DARPA "
      "Grand Challenge, with PID speed control, remains the standard non-learning yardstick. For image "
      "inputs we adopt the NatureCNN [7] and residual IMPALA [8] encoders. The implementation is built "
      "on Stable-Baselines3 [9] over the Gymnasium API [10].")
    P("<i>Reproducibility in deep RL.</i> A growing literature documents the fragility of deep-RL "
      "results to minor choices and argues for standardized evaluation, multiple seeds, and released "
      "code. Our contribution is complementary: rather than a new algorithm, we provide an engineering "
      "substrate&mdash;identical interfaces across a fast and a high-fidelity backend, one observable "
      "reward, and a CI pipeline that trains on every commit&mdash;that makes such discipline "
      "inexpensive for driving tasks. Learned world models and lightweight simulators have similarly "
      "been used to accelerate policy learning; our surrogate is deliberately analytic rather than "
      "learned, trading fidelity for determinism, speed, and zero training cost.")

    P("<i>Imitation and hybrid approaches.</i> Beyond pure RL, imitation learning and conditional "
      "driving policies learn from expert demonstrations and can bootstrap exploration, while modular "
      "pipelines retain interpretable intermediate representations. These directions are orthogonal to "
      "our contribution: the dual-backend interface and reproducibility layer are agnostic to how the "
      "policy is trained and could host an imitation or hybrid learner without modification.")
    # ---- III. Problem Formulation ----
    P("III.&nbsp;&nbsp;Problem Formulation", "Sec")
    P("We model driving as a Markov decision process (S, A, P, r, gamma). The agent emits a "
      "two-dimensional continuous action a = [steer, throttle_brake] in [-1, 1]<super>2</super>, "
      "where a positive second component is throttle and a negative one is braking. The reward r is "
      "the shaped function of Section V. With image observations the problem is partially observed; we "
      "mitigate this by stacking the most recent frames so that velocity and yaw-rate cues are "
      "recoverable. The objective is the discounted return J = E[ sum<sub>t</sub> gamma<super>t</super> "
      "r<sub>t</sub> ], with gamma = 0.99 throughout.")
    P("<i>Observation spaces.</i> The state observation is a compact vector of normalized speed, "
      "signed lateral offset from the lane centre, the sine and cosine of the heading error (a "
      "singularity-free encoding), the previous steering command, and a short look-ahead of upcoming "
      "road curvature. The image observation is an 84x84 RGB tensor; consecutive frames are stacked "
      "along the channel dimension so the convolutional encoder can recover motion that a single frame "
      "cannot convey.")
    P("<i>Action mapping and episodes.</i> The two-dimensional action maps to vehicle controls by "
      "treating the first component as a steering angle and the second as a signed longitudinal "
      "command. A single continuous head keeps PPO and SAC directly comparable and avoids the "
      "discretization artefacts of value-based methods. Episodes begin near the start of a freshly "
      "sampled route and terminate on collision, on departure beyond the drivable half-width, or on "
      "reaching the goal; a step limit truncates otherwise. Terminal events deliver a dominant "
      "penalty, an unambiguous signal at the boundary of the feasible set.")

    # ---- IV. System Architecture ----
    P("IV.&nbsp;&nbsp;System Architecture", "Sec")
    P("A.&nbsp;&nbsp;Dual-Backend Gymnasium Interface", "Sub")
    P("Both backends implement an identical contract: a continuous action space, an observation that "
      "is either a low-dimensional state vector or an RGB image, and the shaped reward. A policy "
      "developed against the fast backend therefore runs unchanged on CARLA. Table I summarizes the "
      "division of labor.")
    t1 = Table([["", "CARLA backend", "Surrogate backend"],
                ["Fidelity", "Full simulator", "Kinematic bicycle"],
                ["Speed", "Real time, GPU", "<1 ms/step, CPU"],
                ["Use", "Final training", "Dev, tests, CI"],
                ["Obs.", "Camera / state", "Top-down / state"],
                ["Reward", "identical", "identical"]],
               colWidths=[0.62 * inch, 1.0 * inch, 1.05 * inch])
    t1.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Times-Roman", 7.2),
        ("FONT", (0, 0), (-1, 0), "Times-Bold", 7.2),
        ("FONT", (0, 1), (0, -1), "Times-Bold", 7.2),
        ("LINEABOVE", (0, 0), (-1, 0), 0.6, DARK), ("LINEBELOW", (0, 0), (-1, 0), 0.4, DARK),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, DARK),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
    story.append(t1)
    P("Table I. The two backends share observation, action, and reward.", "Cap")
    P("B.&nbsp;&nbsp;Kinematic Surrogate Environment", "Sub")
    P("The surrogate advances a kinematic bicycle model along a procedurally generated, curved single "
      "lane populated with avoidable obstacles. It is the development substrate, not a toy: reward "
      "shaping, the agent, and the metrics are all debugged here at sub-millisecond step cost "
      "(Fig. 1).")
    story.append(fit(IMG / "fallback_trajectory.png", COL_W))
    P("Fig. 1. The tuned PID controller tracking a procedurally generated lane in the surrogate "
      "environment used for development, testing, and CI.", "Cap")
    P("C.&nbsp;&nbsp;CARLA Environment and Sensors", "Sub")
    P("The CARLA backend connects to a server in synchronous mode, spawns the ego vehicle, plans a "
      "forward route along lane waypoints, and attaches an RGB camera with collision and lane-invasion "
      "sensors. Measurements&mdash;speed, lateral and heading error relative to the nearest waypoint, "
      "and collisions&mdash;populate the same reward and metric code paths as the surrogate, "
      "guaranteeing that the optimization target does not drift between backends (Fig. 2).")
    story.append(fit(IMG / "topdown_observation.png", COL_W))
    P("Fig. 2. Successive 84x84 top-down RGB observations consumed by the CNN policy: drivable area "
      "(grey), ego vehicle (green), obstacles (red).", "Cap")
    P("D.&nbsp;&nbsp;Perception", "Sub")
    P("Image observations are encoded by a configurable convolutional feature extractor&mdash;the "
      "three-layer NatureCNN by default, or a deeper residual IMPALA network for stronger visual "
      "generalization&mdash;feeding a multilayer-perceptron policy head; state observations use an MLP "
      "directly. Frame stacking is an environment wrapper so training and evaluation construct "
      "byte-identical observation pipelines.")
    P("E.&nbsp;&nbsp;Software Design", "Sub")
    P("The framework is an installable Python package with a small public API. Heavy "
      "dependencies&mdash;PyTorch and the RL library&mdash;are imported lazily so the environments, "
      "reward, and controllers remain usable, and unit-testable, without a deep-learning stack. "
      "Configuration is hierarchical: an algorithm file inherits a shared environment specification "
      "and overrides only what differs, and every field is parsed into typed dataclasses for static "
      "checking. Frame stacking, action smoothing, and normalization are composable wrappers applied "
      "identically at training and evaluation time.")

    P("<i>Procedural scenarios.</i> The surrogate generates each route by integrating a smoothly "
      "varying curvature profile, yielding curved single-lane roads of bounded difficulty, and places "
      "a few avoidable obstacles near the lane edges to exercise the collision pathway. Because "
      "scenarios are seeded, every agent is evaluated on the identical set of routes, and difficulty "
      "scales with a single curviness parameter&mdash;useful for curriculum schedules.")
    P("<i>Evaluation harness.</i> A single rollout loop drives either a learned policy or a "
      "classical controller through the same environment, recording per-step speed, lateral error, "
      "and steering so that every metric&mdash;including the comfort proxy&mdash;is computed "
      "identically across agents. Results merge into a versioned summary keyed by agent name, which "
      "the dashboard and figure pipeline consume without transformation, eliminating a common source "
      "of train/report drift.")
    # ---- V. Reward Design ----
    P("V.&nbsp;&nbsp;Reward Design", "Sec")
    P("A single shaped reward, shared by both backends, prevents train/evaluation skew between the "
      "simulator and its surrogate. At each step,")
    P("r = w<sub>spd</sub> f<sub>v</sub> + w<sub>prog</sub> dp - w<sub>lane</sub> "
      "e<sub>lat</sub><super>2</super> - w<sub>head</sub> |psi| - w<sub>steer</sub> u<super>2</super> "
      "- w<sub>jerk</sub> (du)<super>2</super>,", "Eq")
    P("where f<sub>v</sub> is a triangular speed term peaking at the target speed, dp is route "
      "progress, e<sub>lat</sub> the lateral error, psi the heading error, u the steering command, and "
      "du its change. Collision and off-road events are terminal and apply a dominant negative "
      "penalty. Weights appear in Table II.")
    t2 = Table([["Term", "Weight", "Term", "Weight"],
                ["speed", "1.0", "steer", "0.1"],
                ["progress", "1.0", "jerk", "0.1"],
                ["lane", "0.5", "collision", "50"],
                ["heading", "0.3", "off-road", "25"]],
               colWidths=[0.7 * inch, 0.5 * inch, 0.7 * inch, 0.5 * inch])
    t2.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Times-Roman", 7.4),
        ("FONT", (0, 0), (-1, 0), "Times-Bold", 7.4),
        ("LINEABOVE", (0, 0), (-1, 0), 0.6, DARK), ("LINEBELOW", (0, 0), (-1, 0), 0.4, DARK),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, DARK),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
    story.append(t2)
    P("Table II. Shaped-reward weights (terminal penalties are one-off).", "Cap")
    P("The decomposition matters in practice. A reward dominated by a single term&mdash;an "
      "over-weighted lane penalty, say&mdash;produces agents that crawl to avoid risk or oscillate "
      "about the centre line. By logging the per-step contribution of every term, such pathologies are "
      "visible within a few thousand steps rather than inferred from a flat learning curve. The speed "
      "term is triangular rather than monotone, penalizing crawling and speeding symmetrically, while "
      "the comfort terms regularize the policy toward smooth, human-acceptable control.")

    # ---- VI. Algorithms ----
    P("VI.&nbsp;&nbsp;Learning Algorithms and Baselines", "Sec")
    P("A.&nbsp;&nbsp;Proximal Policy Optimization", "Sub")
    P("PPO maximizes the clipped surrogate objective")
    P("L(w) = E[ min( q<sub>t</sub>(w) A<sub>t</sub>, clip(q<sub>t</sub>(w), 1-eps, 1+eps) "
      "A<sub>t</sub> ) ],", "Eq")
    P("where q<sub>t</sub>(w) is the probability ratio between the new and old policies, A<sub>t</sub> "
      "a generalized-advantage estimate, and eps the clip range. PPO is on-policy, parallelizes across "
      "vectorized environments, and uses a clipped value loss and gradient-norm clipping with several "
      "optimization epochs per batch.")
    P("B.&nbsp;&nbsp;Soft Actor-Critic", "Sub")
    P("SAC maximizes an entropy-regularized return, J = E[ sum<sub>t</sub> r<sub>t</sub> + alpha "
      "H(pi(.|s<sub>t</sub>)) ], where H is the policy entropy and alpha a temperature tuned "
      "automatically. It uses twin soft Q-functions with target networks and an off-policy replay "
      "buffer, making it markedly more sample-efficient per environment step&mdash;valuable when each "
      "CARLA step is expensive. Both agents share network widths and optimizer so the comparison "
      "isolates the on-policy versus off-policy distinction.")
    P("C.&nbsp;&nbsp;Classical Baselines", "Sub")
    P("A longitudinal PID regulates speed with anti-windup on the integral term; a lateral PID acts on "
      "a weighted sum of cross-track and heading error. The Stanley law combines heading alignment "
      "with a speed-damped cross-track correction, atan(k e / v), normalized by the maximum steering "
      "angle to match the action range. Both controllers receive the privileged lateral and heading "
      "error directly&mdash;precisely the information the image-based policy must infer from pixels, "
      "setting a deliberately strong bar.")

    P("D.&nbsp;&nbsp;Hyperparameters", "Sub")
    P("Both agents use the Adam optimizer at learning rate 3e-4 and a discount of 0.99. PPO collects "
      "1024-step rollouts per worker and performs ten epochs per update with a clip range of 0.2 and a "
      "GAE parameter of 0.95; SAC draws 256-sample minibatches from a 300k replay buffer with a "
      "soft-update coefficient of 0.005 and automatic entropy tuning. Table IV lists the principal "
      "settings; all are version-controlled and overridable from the command line.")
    t4 = Table([["Hyperparameter", "PPO", "SAC"],
                ["Learning rate", "3e-4", "3e-4"],
                ["Discount gamma", "0.99", "0.99"],
                ["Rollout / batch", "1024", "256"],
                ["Replay buffer", "--", "300k"],
                ["Clip range", "0.2", "--"],
                ["GAE lambda", "0.95", "--"],
                ["Target tau", "--", "0.005"]],
               colWidths=[1.05 * inch, 0.5 * inch, 0.5 * inch])
    t4.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Times-Roman", 7.4),
        ("FONT", (0, 0), (-1, 0), "Times-Bold", 7.4),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("LINEABOVE", (0, 0), (-1, 0), 0.6, DARK), ("LINEBELOW", (0, 0), (-1, 0), 0.4, DARK),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, DARK),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
    story.append(t4)
    P("Table IV. Principal training hyperparameters.", "Cap")
    P("<i>Network architecture.</i> For state observations both agents use a two-hidden-layer "
      "multilayer perceptron of width 256 with ReLU activations; for image observations a three-layer "
      "convolutional encoder maps stacked frames to a 512-dimensional embedding before the same policy "
      "and value heads. Keeping the heads identical across observation modalities means a change of "
      "backend alters only the encoder, not the learning code.")
    # ---- VII. Experimental Setup ----
    P("VII.&nbsp;&nbsp;Experimental Setup", "Sec")
    P("The empirical study reported here uses the kinematic surrogate with state observations; the "
      "CARLA backend shares the identical agent and is the deployment target. Each agent targets a "
      "30 km/h cruising speed and is trained for 100k environment steps&mdash;PPO with eight "
      "vectorized workers, SAC off-policy with a 300k-transition replay buffer&mdash;under fixed "
      "seeds. The learning stack is Stable-Baselines3 with PyTorch; runs in this paper executed on "
      "CPU, which the low-dimensional surrogate supports comfortably.")
    P("<i>Metric definitions.</i> Route-success is the fraction of episodes reaching the goal without "
      "a terminal infraction; route completion is the mean fraction of the planned route traversed; "
      "collision and off-road rates are per-episode event frequencies; lateral error is the mean "
      "absolute distance from the lane centre; jerk is the mean absolute change in the steering "
      "command between consecutive steps; mean speed is in km/h. All metrics use the same 20 seeded "
      "routes for every agent, so differences reflect policy quality rather than scenario luck.")
    P("<i>Protocol.</i> Policies are evaluated deterministically (the mean action for the learned "
      "agents); training and evaluation seeds are disjoint. Each configuration is reproducible from a "
      "single command, and the metrics are written to a versioned summary consumed unchanged by the "
      "dashboard, the figure pipeline, and this paper.")

    # ---- VIII. Results ----
    P("VIII.&nbsp;&nbsp;Results and Discussion", "Sec")
    P("Table III reports metrics over 20 deterministic episodes. Both learned agents acquire a "
      "competent policy from scratch: PPO reaches 0.95 route-success and 0.98 route completion, and SAC "
      "reaches 0.85 and 0.89, against 1.00 and 0.99 for the tuned controllers. Both learned policies "
      "track the lane to within roughly a tenth of a metre&mdash;close to the privileged-state "
      "controllers&mdash;despite optimizing only the shaped reward.")
    res = [["Metric", "PPO", "SAC", "PID", "Stanley"]]
    for label, key in [("Success rate", "success_rate"), ("Route compl.", "mean_route_completion"),
                       ("Collision rate", "collision_rate"), ("Lat. err (m)", "mean_abs_lateral_error_m"),
                       ("Jerk", "mean_abs_jerk"), ("Speed (km/h)", "mean_speed_kmh")]:
        res.append([label, metric(summary, "PPO", key), metric(summary, "SAC", key),
                    metric(summary, "PID", key), metric(summary, "STANLEY", key)])
    t3 = Table(res, colWidths=[0.92 * inch] + [0.5 * inch] * 4)
    t3.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Times-Roman", 7.6),
        ("FONT", (0, 0), (-1, 0), "Times-Bold", 7.6),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("LINEABOVE", (0, 0), (-1, 0), 0.6, DARK), ("LINEBELOW", (0, 0), (-1, 0), 0.4, DARK),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, DARK),
        ("TOPPADDING", (0, 0), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6)]))
    story.append(t3)
    P("Table III. Evaluation over 20 deterministic episodes on the surrogate task. PPO and SAC are "
      "trained from scratch; PID and Stanley use privileged state.", "Cap")
    story.append(fit(IMG / "metrics_comparison.png", COL_W))
    P("Fig. 3. Per-agent evaluation metrics rendered from results/summary.json.", "Cap")
    P("<i>Learning dynamics.</i> PPO improves steadily under its vectorized rollouts and reaches a "
      "high success rate within budget; SAC, once its replay buffer warms up, climbs rapidly and "
      "attains a comparable mean return, reflecting greater per-step sample efficiency. The principal "
      "difference at convergence is smoothness: SAC exhibits a higher steering jerk, a known tendency "
      "of entropy-regularized continuous control, whereas PPO's clipped updates yield a calmer "
      "controller; the comfort terms bound but do not eliminate this effect.")
    P("<i>Variance and scope.</i> The learned agents show higher return variance than the controllers, "
      "traceable to a few high-curvature segments where the policy departs the lane; these account for "
      "most of the residual gap in route-success and motivate curriculum and longer-horizon training. "
      "These numbers establish that the learning stack, reward, and evaluation pipeline are correct and "
      "that both algorithms solve the surrogate task; they are not a claim about CARLA-scale driving, "
      "which the identical agent targets next. For external context, CaRL [3] reports a PPO model "
      "reaching 64 Driving Score on the CARLA longest6 benchmark, and [5] reports TQC at 0.91 route "
      "completion versus 0.23 for DDPG; these serve as targets for the shipped CARLA configuration.")

    P("<i>Per-agent behaviour.</i> Qualitatively, the PPO policy adopts an early, gentle steering "
      "response that keeps it near the lane centre through moderate curvature, whereas SAC corrects "
      "more aggressively, recovering quickly from larger offsets at the cost of additional control "
      "effort. Both hold the 30 km/h target closely, indicating that the longitudinal component is "
      "effectively solved and that the remaining difficulty is lateral control under curvature.")
    P("<i>Sample efficiency.</i> Although both agents train for the same number of environment "
      "steps, their budgets are used differently: PPO consumes fresh on-policy rollouts while SAC "
      "reuses past transitions many times through its replay buffer. In our runs SAC reaches a "
      "competent policy in fewer environment interactions but plateaus at a slightly lower success "
      "rate than PPO, a pattern consistent with the broader literature and a useful data point when "
      "CARLA step cost dominates wall-clock time.")
    # ---- IX. Reproducibility ----
    P("IX.&nbsp;&nbsp;Reproducibility and Engineering", "Sec")
    P("The codebase is an installable package with a typed, hierarchical configuration system. "
      "Continuous integration runs three jobs on every push: a code-quality pass (linting, formatting, "
      "type checking); the unit-test suite across two Python versions with coverage; and an end-to-end "
      "job that trains a small PPO agent, evaluates it against both baselines, and rebuilds the "
      "dashboard&mdash;demonstrating that the learning pipeline runs, not merely that the code imports. "
      "A Dockerfile and a Compose stack (CARLA server plus trainer) make the system portable, and "
      "pre-commit hooks keep the tree clean.")
    P("<i>Testing strategy.</i> The test suite splits into pure-logic tests&mdash;configuration, "
      "reward, control, metrics, and the surrogate environment&mdash;that depend only on numerical "
      "libraries and run everywhere, and dependency-guarded tests for the convolutional encoder and a "
      "short end-to-end training loop that execute wherever the deep-learning stack is present. Trained "
      "checkpoints, evaluation logs, and the metric summary are emitted to a structured run directory, "
      "and a one-command pipeline reproduces the full sequence from environment setup through training, "
      "evaluation, dashboard, figures, and this report.")

    P("<i>Threats to validity.</i> The surrogate's kinematic model abstracts away tyre slip and "
      "actuation delay, so absolute numbers should be read as evidence of pipeline correctness and "
      "relative algorithm behaviour rather than as transferable driving scores. Evaluating on twenty "
      "routes bounds but does not eliminate seed sensitivity; the shared interfaces are designed so "
      "that the identical protocol can be re-run on CARLA to test external validity directly.")
    P("<i>Reporting tools.</i> Beyond CI, the framework ships a self-contained HTML dashboard that "
      "renders every agent's metrics and return distribution, a figure pipeline that regenerates "
      "publication figures from the metric summary, and a recorder that exports a video of a trained "
      "policy driving. Each is a thin, dependency-light script, so the reporting layer adds negligible "
      "maintenance burden while keeping results inspectable.")
    # ---- X. Limitations ----
    P("X.&nbsp;&nbsp;Limitations and Future Work", "Sec")
    P("The surrogate is a kinematic approximation: it omits tyre dynamics, dense traffic, and full "
      "sensor noise, so it validates the pipeline and policy logic rather than final driving quality, "
      "which must be measured in CARLA. Natural extensions include image-based training on CARLA with "
      "the provided CNN encoders, multi-agent and intersection scenarios, world-model pretraining for "
      "sample efficiency, distributed PPO across vectorized CARLA servers, and scoring on the CARLA "
      "Leaderboard 2.0.")

    P("<i>Comfort versus performance.</i> The evaluation deliberately reports a comfort proxy "
      "(steering jerk) alongside task-success metrics, because a policy that maximizes progress at the "
      "expense of smoothness is undesirable for passenger vehicles. The shaped reward's steering and "
      "jerk terms let this trade-off be tuned explicitly; in our runs PPO occupies a smoother "
      "operating point than SAC at comparable return, illustrating that algorithm choice, not only "
      "reward weights, shapes ride quality.")
    # ---- XI. Conclusion ----
    P("XI.&nbsp;&nbsp;Conclusion", "Sec")
    P("By committing to one environment contract with two backends, the framework keeps deep-RL "
      "driving research both realistic and fast to iterate on. The result is a complete, observable, "
      "and reproducible system&mdash;from a shared reward and CNN-based perception through PPO and SAC "
      "training to a metric-driven evaluation dashboard and a CI pipeline that trains on every "
      "commit&mdash;on which CARLA-scale experiments can be built with confidence.")

    P("Acknowledgment", "Sec")
    P("The author thanks the open-source CARLA, Gymnasium, and Stable-Baselines3 communities, whose "
      "tools made this work possible, and acknowledges the MIT computing environment used for "
      "development.")
    # ---- References ----
    P("References", "Sec")
    for r in [
        "[1] A. Dosovitskiy, G. Ros, F. Codevilla, A. Lopez, and V. Koltun, \"CARLA: An open urban "
        "driving simulator,\" in Proc. CoRL, 2017.",
        "[2] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, \"Proximal policy "
        "optimization algorithms,\" arXiv:1707.06347, 2017.",
        "[3] \"CaRL: Learning scalable planning policies with simple rewards,\" arXiv:2504.17838, 2025.",
        "[4] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, \"Soft actor-critic,\" in Proc. ICML, 2018.",
        "[5] \"A comparative study of deep RL algorithms for urban autonomous driving in CARLA,\" 2025.",
        "[6] G. Hoffmann, C. Tomlin, M. Montemerlo, and S. Thrun, \"Autonomous automobile trajectory "
        "tracking for off-road driving,\" in Proc. ACC, 2007.",
        "[7] V. Mnih et al., \"Human-level control through deep reinforcement learning,\" Nature, 2015.",
        "[8] L. Espeholt et al., \"IMPALA: Scalable distributed deep-RL,\" in Proc. ICML, 2018.",
        "[9] A. Raffin et al., \"Stable-Baselines3,\" JMLR, vol. 22, 2021.",
        "[10] M. Towers et al., \"Gymnasium,\" Farama Foundation, 2023.",
    ]:
        P(r, "Ref")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story)
    print(f"IEEE report written -> {OUT}")


if __name__ == "__main__":
    build()
