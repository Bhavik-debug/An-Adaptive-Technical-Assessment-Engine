%% tietreport.cls
%%
%% Local reconstruction of the TIET project-report class used by the
%% Overleaf template ("tietreport"). It reproduces the compiled template's
%% look -- letter paper, book-style right-aligned chapter heads with a rule,
%% alternating running headers, and the same title-page layout -- so that
%% report.tex builds byte-for-byte locally with tectonic/pdflatex.
%%
%% If the project is compiled on Overleaf where the official class already
%% exists, do NOT upload this file; the official one takes precedence.
%%
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{tietreport}[2026/09/16 TIET project report (local reconstruction)]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}
\ProcessOptions\relax
\LoadClass[11pt,letterpaper,twoside,openany]{book}

\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage{lmodern}
\RequirePackage{microtype}
\RequirePackage[letterpaper,top=1.0in,bottom=1.0in,inner=1.05in,outer=0.95in,
  headheight=14pt,headsep=16pt]{geometry}
\RequirePackage{fancyhdr}
\RequirePackage{titlesec}
\RequirePackage{setspace}
\RequirePackage{xcolor}
\RequirePackage{enumitem}
\RequirePackage{caption}
\RequirePackage{hyperref}

\setlist{itemsep=2pt,topsep=4pt,parsep=0pt}
\captionsetup{font=small,labelfont=bf,width=0.92\linewidth}
\setlength{\parskip}{0pt}
\setlength{\parindent}{15pt}
\setlength{\emergencystretch}{3em}
\raggedbottom
\RequirePackage{needspace}
% top-align float-only pages instead of centring them
\setlength{\@fptop}{0pt}\setlength{\@fpbot}{0pt plus 1fil}
\setstretch{1.0}

%% ---------------------------------------------------------------
%% Title-page data commands (the public interface used by report.tex)
%% ---------------------------------------------------------------
\newcommand{\@TitlePageHeader}{Course Code - Course Title}
\newcommand{\@ProjectTitle}{Project Title}
\newcommand{\@ProjectSubTitle}{}
\newcommand{\@TitlePageSubText}{}
\newcommand{\@AdvisorName}{}
\newcommand{\@TitlePageFooterText}{}
\newcommand{\TitlePageHeader}[1]{\renewcommand{\@TitlePageHeader}{#1}}
\newcommand{\ProjectTitle}[1]{\renewcommand{\@ProjectTitle}{#1}}
\newcommand{\ProjectSubTitle}[1]{\renewcommand{\@ProjectSubTitle}{#1}}
\newcommand{\TitlePageSubText}[1]{\renewcommand{\@TitlePageSubText}{#1}}
\newcommand{\AdvisorName}[1]{\renewcommand{\@AdvisorName}{#1}}
\newcommand{\TitlePageFooterText}[1]{\renewcommand{\@TitlePageFooterText}{#1}}

%% ---------------------------------------------------------------
%% Running headers: even pages "N  <course>", odd pages "<title>  N"
%% ---------------------------------------------------------------
\fancypagestyle{tiet}{%
  \fancyhf{}%
  \fancyhead[LE]{\thepage\quad \@TitlePageHeader}%
  \fancyhead[RO]{\@ProjectTitle\quad \thepage}%
  \renewcommand{\headrulewidth}{0.4pt}%
  \renewcommand{\footrulewidth}{0pt}%
}
\fancypagestyle{plain}{%
  \fancyhf{}%
  \renewcommand{\headrulewidth}{0pt}%
  \renewcommand{\footrulewidth}{0pt}%
}
\pagestyle{tiet}

%% ---------------------------------------------------------------
%% Chapter heading: "Chapter N" then the title, right aligned, rule below
%% ---------------------------------------------------------------
\titleformat{\chapter}[display]
  {\normalfont\bfseries\filleft}
  {\Large Chapter \thechapter}
  {6pt}
  {\Huge}
  [\vspace{4pt}\titlerule]
\titleformat{name=\chapter,numberless}[display]
  {\normalfont\bfseries\filleft}
  {}
  {0pt}
  {\Huge}
  [\vspace{4pt}\titlerule]
\titlespacing*{\chapter}{0pt}{8pt}{22pt}
\titlespacing*{name=\chapter,numberless}{0pt}{8pt}{22pt}

\titleformat{\section}{\normalfont\Large\bfseries}{\thesection}{1em}{}
\titleformat{\subsection}{\normalfont\large\bfseries}{\thesubsection}{1em}{}
\titleformat{\subsubsection}{\normalfont\normalsize\bfseries}{\thesubsubsection}{1em}{}
\titlespacing*{\section}{0pt}{3.2ex plus 1ex minus .2ex}{1.8ex plus .2ex}
\titlespacing*{\subsection}{0pt}{2.8ex plus 1ex minus .2ex}{1.2ex plus .2ex}
\titlespacing*{\subsubsection}{0pt}{2.4ex plus 1ex minus .2ex}{1ex plus .2ex}

\setcounter{tocdepth}{2}
\setcounter{secnumdepth}{3}

%% ---------------------------------------------------------------
%% Title page
%% ---------------------------------------------------------------
\renewcommand{\maketitle}{%
  \begin{titlepage}
    \thispagestyle{empty}
    \centering
    \vspace*{1.6cm}
    {\large\bfseries \@TitlePageHeader\par}
    \vspace{1.9cm}
    {\Huge\bfseries \@ProjectTitle\par}
    \vspace{0.55cm}
    {\large \parbox{0.86\textwidth}{\centering \@ProjectSubTitle}\par}
    \vspace{2.6cm}
    {\Large\bfseries Submitted by:\par}
    \vspace{0.25cm}
    {\large \@author\par}
    \vspace{0.55cm}
    {\normalsize \@TitlePageSubText\par}
    \vspace{2.4cm}
    {\Large\bfseries Submitted to:\par}
    \vspace{0.25cm}
    {\large \@AdvisorName\par}
    \vfill
    {\normalsize \@TitlePageFooterText\par}
    \vspace{0.6cm}
  \end{titlepage}
  \clearpage
}

\endinput
