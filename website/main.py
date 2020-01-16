from flask import Flask, send_from_directory, session
from flask_session import Session
from flask import render_template
from wtforms import (Form, TextAreaField, validators, SubmitField)
from flask import request, flash
import os
import torch


# import my models here
import sys
sys.path.append("..")
sys.path.append("../code_summary")
# generation
from code_generation.eval import generation_pre_setup, generate
from code_sentiment.eval import sentiment_pre_setup, analysis_sentiment
from code_useful.tools import load_model as load_useful_model
from code_useful.tools import analyze_useful
from onmt.bin.translate import load_translator, translate

global_device = torch.device("cpu")
if torch.cuda.is_available():
    global_device = torch.device("cuda")

# prepare model
generation_model, generation_tokenizer = generation_pre_setup(
    # checkpoint_name="1579061867",
    # checkpoint_name="1579079968",
    checkpoint_name="1579140567",
    device=global_device,
    pretrained_bert_name="gpt2",
    model_file="model_params.model"
)

sentiment_model, sentiment_tokenizer = sentiment_pre_setup(
    checkpoint_name="1578973534",
    device=global_device,
    pretrained_bert_name="bert-base-uncased",
    model_file="model_params.model"
)

useful_model = load_useful_model()


summary_model = load_translator()

def write_score(predict_label, text_list, weight_list):
    # '<span style="background-color:rgba(255,0,0,1.0);">xxx</span>'
    if predict_label == 1:
        sentiment_predict_str = '<span style="background-color:rgba(255,0,0,1.0);">Positive</span>'
        str_1 = '<span style="background-color:rgba(255,0,0,'
    else:
        sentiment_predict_str = '<span style="background-color:rgba(0,0,255,1.0);">Negative</span>'
        str_1 = '<span style="background-color:rgba(0,0,255,'

    assert len(text_list) == len(weight_list)

    show_html_code = ""
    max_weight = float(max(weight_list))
    # print("max_weight", max_weight)
    for idx, word in enumerate(text_list):
        # print(word, weight_list[idx])
        ww = 0.0
        if weight_list[idx] > 0.0:
            ww = weight_list[idx] / max_weight
        this_str = str_1 + str(ww) + ');">{}</span> '.format(str(word))
        show_html_code += this_str
    return [sentiment_predict_str, show_html_code]


def write_sentence(outputs):
    if outputs[0] == 1:
        useful_predict_str = '<span style="background-color:rgba(255,0,0,1.0);">Useful</span>'
    else:
        useful_predict_str = '<span style="background-color:rgba(0,0,255,1.0);">Useless</span>'
    str_2 = '<span style="background-color:rgba(255,0,0,'
    str_1 = '<span style="background-color:rgba(0,0,255,'

    max_len = len(outputs[1])
    show_html_code = ""
    for index, item in enumerate(outputs[1]):
        sentence = item[0]
        if index == 0:  # delete "[CLS]"
            sentence = sentence[5:]
        if index == max_len - 1:  # delete "[SEP]"
            sentence = sentence[:-5]
        sentence = sentence.strip()

        i_useless = item[1]
        i_useful = item[2]
        i_value = i_useless - i_useful
        if i_value > 0:
            this_str = str_1 + str(i_value) + ');">{}</span> '.format(str(sentence))
        else:
            this_str = str_2 + str(-i_value) + ');">{}</span> '.format(str(sentence))
        show_html_code += this_str
    return [useful_predict_str, show_html_code]


app = Flask(__name__, template_folder='./', static_url_path='')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = os.urandom(24)
Session(app)


class ReusableForm(Form):
    """User entry form for entering specifics for generation"""
    review_text = TextAreaField(
        "Review Text:",
        validators=[
            validators.InputRequired(),
            validators.length(max=1000),
        ],
        render_kw={'class': 'form-control', 'rows': 5, 'placeholder': u'Hi, write something here!'},
    )

    # Submit button
    generate_stats = SubmitField(
        'Generate',
        render_kw={'class': 'btn btn-success col-md-4',},
    )
    analyze_stats = SubmitField(
        'Analyze',
        render_kw={'class': 'btn btn-info col-md-4', },
    )


@app.route('/static/<path:path>')
def send_js(path):
    return send_from_directory('./static/', path)


# Home page
@app.route("/", methods=['GET', 'POST'])
def home():
    """Home page of app with form"""
    # Create form
    form = ReusableForm(request.form)

    # User action
    if request.method == 'POST':
        # print('Hello: {}'.format("Here is yes"))
        if form.generate_stats.data:
            review_text = form.review_text.data.strip()
            if review_text != "":
                predict_word = generate(review_text, generation_model, generation_tokenizer, device=global_device)
                review_text = review_text + ' ' + predict_word
                form.review_text.data = review_text

        elif form.analyze_stats.data:
            review_text = form.review_text.data.strip()
            if review_text != "":
                print("review_text:\n", review_text)
                # Sentiment
                [predict_label, token_list, important_list] = analysis_sentiment(review_text, sentiment_model, sentiment_tokenizer, global_device)
                [sentiment_predict_str, sentiment_html_code] = write_score(predict_label, token_list, important_list)
                analysis_sentiment_html_code = "Sentiment: " + sentiment_predict_str + '<br>' + sentiment_html_code
                print("Sentiment: \n", sentiment_predict_str)

                # Useful
                useful_outputs = analyze_useful(useful_model, review_text)
                # print("useful_outputs:\n", useful_outputs)
                [useful_predict_str, useful_html_code] = write_sentence(useful_outputs)
                print("Usefulness:\n", useful_predict_str)
                analysis_useful_html_code = "Usefulness: " + useful_predict_str + '<br>' + useful_html_code

                # Summary
                summary_text = translate(review_text, summary_model)
                print("summary:\n", summary_text)
                analysis_summary = "Summary: " + summary_text
                return render_template('index.html', form=form, analysis_sentiment=analysis_sentiment_html_code,
                                       analysis_summary=analysis_summary,
                                       analysis_useful=analysis_useful_html_code)

    # Send template information to index.html
    return render_template('index.html', form=form)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)

