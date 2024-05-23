## EsportsBench
The EsportsBench datasets are meant to facilitate research and experimentation on real world competition data spanning many years of competitions including a diverse range of genres and competition formats.

## Licenses
This *code* is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/


### Setup
To collect the data yourself, you will need to obtain Aliculac and Liquipedia LPDB API key. Aligulac keys can be generated at [http://aligulac.com/about/api/](http://aligulac.com/about/api/) and Liquipedia LPDB keys can be requested in the Liquipedia [discord server](https://discord.gg/hW3T8BQr).

Add your key(s) to `.dotenv-template` and rename it to `.env` so that the data pipelines can access them.

### Data Licences
The data collected by these pipelines is collected from different sources with their own licenses. If you reproduce the the data collection and experiments understand the retrieved data and results falls under those licenses.

The StarCraft II data is from [Aligulac](http://aligulac.com/)

The League of Legends data is from [Leaguepedia](https://lol.fandom.com/) under a [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)

The data for all other games is from [Liquipedia](https://liquipedia.net/) under a [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)
