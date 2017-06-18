import pixivpy3 as pixiv
import src.download_settings as settings

papi = pixiv.PixivAPI()
papi.login(settings.id, settings.password)

def download(dst_path):
    for artist in settings.artists:
        user_works = papi.users_works(artist[1], per_page=30000)
        apapi = pixiv.AppPixivAPI()
        print('Artist: ' + user_works.response[0].user.name)
        print('Works: %d' % user_works.pagination.total)
        for work_i in range(user_works.pagination.total):
            illust = user_works.response[work_i]
            print('Title: ' + illust.title)
            apapi.download(illust.image_urls.large, dst_path)
    print('finish download')

if __name__ == '__main__':
    download('./imgs/colors/')
    pass