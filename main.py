from __future__ import print_function
from model import StyleTransferModel
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import os
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext.dispatcher import run_async
import torch
from PIL import Image
import process_stylization
from photo_wct import PhotoWCT
from photo_gif import GIFSmoothing

model = StyleTransferModel()
p_wct = PhotoWCT()
p_wct.load_state_dict(torch.load('photo_wct.pth'))
p_pro = GIFSmoothing(r=35, eps=0.001)
first_image_file = {}
f = []
f1 = []
pool = ThreadPoolExecutor()


def send_prediction_on_photo(update, context):
    global f
    global f1
    # Нам нужно получить две картинки, чтобы произвести перенос стиля, но каждая картинка приходит в
    # отдельном апдейте, поэтому в простейшем случае мы будем сохранять id первой картинки в память,
    # чтобы, когда уже придет вторая, мы могли загрузить в память уже сами картинки и обработать их.
    # Точно место для улучшения, я бы
    bot = context.bot
    if update.message.text == '/style':
        f.append('1')
        update.message.reply_text(
            "Пришли 2 фотографии, первая фотография - то, что хочешь изменить. Вторая - стиль, который хочешь перенести(любая картина художника).")

    elif len(f) != 0:
        chat_id = update.message.chat_id
        print("Got image from {}".format(chat_id))

        # получаем информацию о картинке
        image_info = update.message.photo[-1]
        image_file = bot.get_file(image_info)

        if chat_id in first_image_file:
            # первая картинка, которая к нам пришла станет content image, а вторая style image
            print("работает style_transfer")
            content_image_stream = BytesIO()
            first_image_file[chat_id].download(out=content_image_stream)
            del first_image_file[chat_id]

            style_image_stream = BytesIO()
            image_file.download(out=style_image_stream)
            output = model.transfer_style(content_image_stream, style_image_stream, num_steps=300)

            # теперь отправим назад фото
            output_stream = BytesIO()  #
            output.save(output_stream, format='PNG')
            output_stream.seek(0)
            bot.send_photo(chat_id, photo=output_stream)
            print("Sent Photo to user")
            f = []
        else:
            first_image_file[chat_id] = image_file  #

    if update.message.text == '/photo_real':
        f1.append('1')
        update.message.reply_text(
            "Пришли 2 фотографии, первая фотография - то, что хочешь изменить. Вторая - стиль, который хочешь перенести(любая фотография).")

    elif len(f1) != 0:
        chat_id = update.message.chat_id
        print("Got image from {}".format(chat_id))

        # получаем информацию о картинке
        image_info = update.message.photo[-1]
        image_file = bot.get_file(image_info)

        if chat_id in first_image_file:
            # первая картинка, которая к нам пришла станет content image, а вторая style image
            print("работает photo_real")
            content_image_stream = BytesIO()
            first_image_file[chat_id].download(out=content_image_stream)
            del first_image_file[chat_id]

            style_image_stream = BytesIO()
            image_file.download(out=style_image_stream)
            process_stylization.stylization(
                stylization_module=p_wct,
                smoothing_module=p_pro,
                content_image_path=content_image_stream,
                style_image_path=style_image_stream,
                content_seg_path=[],
                style_seg_path=[],
                output_image_path='img.jpg',
                cuda=False,
                save_intermediate=False,
                no_post=False
            )
            output = Image.open('img.jpg')
            # теперь отправим назад фото
            output_stream = BytesIO()  #
            output.save(output_stream, format='PNG')
            output_stream.seek(0)
            bot.send_photo(chat_id, photo=output_stream)
            f1 = []
            print("Sent Photo to user")
        else:
            first_image_file[chat_id] = image_file  #


@run_async
def start(update, context):
    reply_keyboard = [['/style', '/photo_real', '/cancel']]

    update.message.reply_text(
        'Привет, это бот, который переносит стиль изображения. Чтобы начать отправьте /style или /photo_real. Пришлите /cancel, чтобы перестать со мной общаться.',
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True))


@run_async
def log(user):
    logger.info("User %s canceled the conversation.", user.first_name)


@run_async
def cancel(update, context):
    user = update.message.from_user
    log(user)
    update.message.reply_text('Пока.',
                              reply_markup=ReplyKeyboardRemove())

    return ConversationHandler.END


@run_async
def unknown(update, context):
    bot = context.bot
    bot.sendMessage(chat_id=update.message.chat.id, text="Извини, но я тебя не понимаю.")


if __name__ == '__main__':
    from telegram.ext import Updater, MessageHandler, Filters, CommandHandler, ConversationHandler
    import logging

    # Включим самый базовый логгинг, чтобы видеть сообщения об ошибках
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO)
    logger = logging.getLogger(__name__)
    # используем прокси, так как без него у меня ничего не работало.
    # если есть проблемы с подключением, то попробуйте убрать прокси или сменить на другой
    # прокси ищется в гугле как "socks4 proxy"
    updater = Updater(token='808687119:AAFa1IUjO7mIyqgdDKHwjbCJZ4e04Z85XjU',
                      request_kwargs={'proxy_url': 'socks5h://163.172.152.192:1080'},
                      use_context=True)

    # В реализации сложных диалогов скорее всего будет удобнее использовать Conversation Handler
    # вместо назначения handler'ов таким способом
    pool.submit(updater)
    start_handler = CommandHandler('start', start)
    updater.dispatcher.add_handler(start_handler)
    pool.submit(
        updater.dispatcher.add_handler(MessageHandler((Filters.photo | Filters.command), send_prediction_on_photo)))
    updater.dispatcher.add_handler(CommandHandler('cancel', cancel))
    unknown_handler = MessageHandler(Filters.command, unknown)
    updater.dispatcher.add_handler(unknown_handler)
    updater.dispatcher.add_handler(MessageHandler(Filters.text, unknown))
    updater.start_polling()
    updater.idle()