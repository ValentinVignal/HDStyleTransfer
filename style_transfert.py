import src
from epicpath import EPath

content_path_list, style_path_list = src.st.data.get_data()

file_combination = src.st.data.get_next_files(content_path_list, style_path_list)


if file_combination is not None:
    # image_couple = src.images.load_content_style_img(content_path.as_posix(), style_path.as_posix(), plot_it=True)
    src.st.var.param.n = file_combination.n

    extractor = src.st.StyleContentModel(
        style_layers=src.st.var.param.style_layers.value,
        content_layers=src.st.var.param.content_layers.value,
        content_gram_layers=src.st.var.param.content_gram_layers.value
    )

    optimizers = src.st.Optimizers(
        shape=(1,),
        lr=src.st.var.param.lr.value
    )

    file_combination.results_folder.mkdir()
    parameters_path = EPath('results/parameters.txt')
    if not parameters_path.exists():
        src.st.var.param.save_all_txt(parameters_path)
    p_path = EPath(f'results/p{src.st.var.param.n}.txt')
    if not p_path.exists():
        src.st.var.param.save_current_txt(p_path)

    src.st.train.style_transfert(
        file_combination=file_combination,
        extractor=extractor,
        optimizers=optimizers,
        epochs=src.st.var.param.epochs.value,
        steps_per_epoch=src.st.var.param.steps_per_epoch.value
    )




else:
    print('No result_path left...')



