import time
import tqdm
import logging
import discord
import requests
from julia import Main
import ctypes
from multiprocessing import Process, Array, Value

C4 = int('#d62728'.strip('#'), 16)
C3 = int('#2ca02c'.strip('#'), 16)

def create_pbar_discord_embed(pbar, desc = "Simulation Status"):
    embed = discord.Embed(title = f"{desc}: Running", color = C4)
    embed.add_field(name = "Progress", value = pbar)
    return embed

def create_pbar_discord(total, desc = "Simulation Status"):
    pbar = tqdm.tqdm(total = total, ncols = 60, disable = False)
    pbar.disable = True
    embed = create_pbar_discord_embed(pbar, desc)
    return pbar, embed


def discord_progressbar(pbar_str, webhook_url, msg_id, fname_counter, total, desc = None, dt = 2):
    pbar, embed = create_pbar_discord(total, desc)
    webhook = discord.Webhook.from_url(webhook_url, adapter=discord.RequestsWebhookAdapter())
    msg = webhook.send(embed = embed, wait = True)
    msg_id.value = msg.id
    embed_dict = embed.to_dict()
    while True:
        try:
            with open(fname_counter, 'r') as f:
                for line in f:
                    pass
                cnt = int(line.strip('\n'))
        except Exception as e:
            raise e
        try:
            webhook = discord.Webhook.from_url(webhook_url, adapter=discord.RequestsWebhookAdapter())
            if (cnt != pbar.n) or (cnt == 0):
                pbar.n = cnt
                embed_dict['fields'][0]['value'] = str(pbar)
                webhook.edit_message(message_id=msg_id.value, embed = discord.Embed.from_dict(embed_dict))
                pbar_str[:len(str(pbar))] = str(pbar)
                if pbar.n >= pbar.total:
                    break
            time.sleep(dt)
        except Exception as e:
            time.sleep(dt)
            logging.warning(e)
            pass
            

def run_simulation(fname_counter, n_trajectories, ensembleproblem = "ens_prob", save_everystep = True):
    Main.n_trajectories = n_trajectories
    Main.eval(f"""
    progress = Progress(n_trajectories, showspeed = true)
    @sync sim = begin
        @async begin
            tasksdone = 0
            while tasksdone < n_trajectories
                tasksdone += take!(channel)
                update!(progress, tasksdone)
                open("{fname_counter}", "w") do io
                    print(io, tasksdone)
                end
            end
        end
        @async begin
            @time global sim = solve({ensembleproblem}, Tsit5(), EnsembleDistributed(), trajectories=n_trajectories,
                        abstol = 5e-7, reltol = 5e-4, callback = cb, save_everystep = {str(save_everystep).lower()})
        end
    end
    """)
            
def start_ensemble_with_progressbar(total, desc, webhook_url, fname_counter, ensembleproblem, save_everystep = True):
    webhook = discord.Webhook.from_url(
        webhook_url, 
        adapter=discord.RequestsWebhookAdapter())
    pbar_str = Array(ctypes.c_wchar, " "*60)
    msg_id = Value(ctypes.c_longlong, 0)
    time.sleep(0.1)
    pthread = Process(target = discord_progressbar, 
                                args = (pbar_str, webhook_url, msg_id, fname_counter, total, desc))
    pthread.start()

    run_simulation(fname_counter, total, ensembleproblem, save_everystep)

    pthread.join()
    
    embed = create_pbar_discord_embed("", desc)
    embed_dict = embed.to_dict()
    embed_dict['fields'][0]['value'] = pbar_str[:]
    embed = discord.Embed.from_dict(embed_dict)
    embed.color = C3
    embed.title = embed.title.replace("Running", "Completed")
    webhook.delete_message(message_id = msg_id.value)
    webhook.send(embed = embed)